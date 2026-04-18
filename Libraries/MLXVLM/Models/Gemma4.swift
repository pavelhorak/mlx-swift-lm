//
//  Gemma4.swift (VLM)
//  mlx-swift-lm
//
//  Gemma 4 Vision-Language Model — uses SigLIP vision tower (shared with Gemma3)
//  + Gemma4TextModel (from MLXLLM) + multimodal projector.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXLLM
import Tokenizers

// MARK: - Configuration

public struct Gemma4VLMConfiguration: Codable, Sendable {
    public let textConfig: Gemma4TextConfiguration
    public let visionConfig: Gemma3VisionConfiguration
    public let modelType: String
    public let mmTokensPerImage: Int
    public let quantization: BaseConfiguration.Quantization?

    private let _vocabularySize: Int?
    private let _padTokenId: Int?

    public var vocabularySize: Int { _vocabularySize ?? textConfig.vocabularySize }
    public var hiddenSize: Int { textConfig.hiddenSize }
    public var padTokenId: Int { _padTokenId ?? 0 }

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case modelType = "model_type"
        case mmTokensPerImage = "mm_tokens_per_image"
        case quantization
        case _vocabularySize = "vocab_size"
        case _padTokenId = "pad_token_id"
    }
}

// MARK: - Multimodal Projector

class Gemma4MultiModalProjector: Module, UnaryLayer {
    @ModuleInfo(key: "mm_input_projection_weight") var mmInputProjectionWeight: MLXArray
    @ModuleInfo(key: "mm_soft_emb_norm") var mmSoftEmbNorm: Gemma.RMSNorm
    @ModuleInfo var avgPool: AvgPool2d

    let patchesPerImage: Int
    let tokensPerSide: Int
    let kernelSize: Int

    init(config: Gemma4VLMConfiguration) {
        self._mmInputProjectionWeight.wrappedValue = ones([
            config.visionConfig.hiddenSize,
            config.textConfig.hiddenSize,
        ])

        self._mmSoftEmbNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.visionConfig.hiddenSize,
            eps: config.visionConfig.layerNormEps
        )

        self.patchesPerImage =
            config.visionConfig.imageSize / config.visionConfig.patchSize
        self.tokensPerSide = Int(sqrt(Double(config.mmTokensPerImage)))
        self.kernelSize = patchesPerImage / tokensPerSide

        self.avgPool = AvgPool2d(
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(kernelSize)
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, _, l) = (x.dim(0), x.dim(1), x.dim(2))

        var reshapedVisionOutputs = x.transposed(0, 2, 1)
        reshapedVisionOutputs = reshapedVisionOutputs.reshaped(
            b, l, patchesPerImage, patchesPerImage)
        reshapedVisionOutputs = reshapedVisionOutputs.transposed(0, 2, 3, 1)

        var pooledVisionOutputs = avgPool(reshapedVisionOutputs)
        pooledVisionOutputs = pooledVisionOutputs.transposed(0, 3, 1, 2).flattened(start: 2)
        pooledVisionOutputs = pooledVisionOutputs.transposed(0, 2, 1)

        let normedVisionOutputs = mmSoftEmbNorm(pooledVisionOutputs)
        let projectedVisionOutputs = einsum(
            "btm,md->btd", normedVisionOutputs, mmInputProjectionWeight)

        return projectedVisionOutputs.asType(x.dtype)
    }
}

// MARK: - Gemma 4 VLM Model

public class Gemma4VLM: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Gemma4TextModel
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: Gemma4MultiModalProjector

    public let config: Gemma4VLMConfiguration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        return languageModel.newCache(parameters: parameters)
    }

    public init(_ config: Gemma4VLMConfiguration) {
        self.config = config
        self._visionTower.wrappedValue = VisionModel(config: config.visionConfig)
        self._languageModel.wrappedValue = Gemma4TextModel(config.textConfig)
        self._multiModalProjector.wrappedValue = Gemma4MultiModalProjector(config: config)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray, pixelValues: MLXArray?, mask: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        guard let pixelValues else {
            return (languageModel.embedTokens(inputIds), nil)
        }

        let inputsEmbeds = languageModel.embedTokens(inputIds)
        let processedPixels = pixelValues.transposed(0, 2, 3, 1).asType(inputsEmbeds.dtype)
        let (hiddenState, _, _) = visionTower(processedPixels, outputHiddenStates: true)
        let imageFeatures = multiModalProjector(hiddenState)

        let (finalEmbedding, finalMask) = prepareInputsForMultimodal(
            imageFeatures: imageFeatures,
            inputsEmbeds: inputsEmbeds,
            inputIds: inputIds,
            attentionMask: mask
        )
        return (finalEmbedding, finalMask)
    }

    private func prepareInputsForMultimodal(
        imageFeatures: MLXArray, inputsEmbeds: MLXArray,
        inputIds: MLXArray, attentionMask: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let embedDim = inputsEmbeds.dim(2)
        let scaledImageFeatures = imageFeatures / sqrt(Float(config.textConfig.hiddenSize))

        var finalEmbedding = inputsEmbeds
        let imageTokenId = 262144
        let padTokenId = config.padTokenId

        let imageMask = MLX.equal(inputIds, MLXArray(imageTokenId))
        let padMask = MLX.equal(inputIds, MLXArray(padTokenId))

        var imageMaskExpanded = expandedDimensions(imageMask, axis: -1)
        imageMaskExpanded = repeated(imageMaskExpanded, count: embedDim, axis: -1)

        var padMaskExpanded = expandedDimensions(padMask, axis: -1)
        padMaskExpanded = repeated(padMaskExpanded, count: embedDim, axis: -1)
        finalEmbedding = MLX.where(
            padMaskExpanded, MLXArray.zeros(like: finalEmbedding), finalEmbedding)

        finalEmbedding = maskedScatter(
            finalEmbedding: finalEmbedding,
            imageMaskExpanded: imageMaskExpanded,
            scaledImageFeatures: scaledImageFeatures)

        var finalAttentionMask4d: MLXArray? = nil
        if let attentionMask {
            let e1 = expandedDimensions(attentionMask, axis: 1)
            let e2 = expandedDimensions(attentionMask, axis: 2)
            finalAttentionMask4d = expandedDimensions(e1 * e2, axis: 1)
        }

        return (finalEmbedding.asType(inputsEmbeds.dtype), finalAttentionMask4d)
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        guard let imagePixels = input.image?.pixels else {
            let logits = languageModel(input.text.tokens, cache: cache)
            return .logits(LMOutput(logits: logits))
        }

        let (inputEmbeddings, _) = getInputEmbeddings(
            inputIds: input.text.tokens, pixelValues: imagePixels, mask: input.text.mask)

        let logits = languageModel.generateWithEmbedding(
            input.text.tokens, inputEmbedding: inputEmbeddings, cache: cache)

        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        return languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = languageModel.sanitize(weights: weights)
        processedWeights = visionTower.sanitize(weights: processedWeights)
        return processedWeights
    }
}

// MARK: - LoRA

extension Gemma4VLM: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.loraLayers
    }
}

// MARK: - Processor

public struct Gemma4Processor: UserInputProcessor {
    private let config: Gemma3ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(config: Gemma3ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Format messages using Gemma chat template
        let messages = Qwen2VLMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext)

        // Process images using same pipeline as Gemma3
        let targetSize = CGSize(width: config.imageSize, height: config.imageSize)
        var pixelValues: MLXArray? = nil

        if !input.images.isEmpty {
            let processedImages = try input.images.map { image in
                let ciImage = try image.asCIImage()
                let processed = MediaProcessing.apply(ciImage, processing: nil)
                let srgb = MediaProcessing.inSRGBToneCurveSpace(processed)
                let resized = MediaProcessing.resampleBicubic(srgb, to: targetSize)
                let normalized = MediaProcessing.normalize(
                    resized, mean: config.imageMeanTuple, std: config.imageStdTuple)
                return MediaProcessing.asMLXArray(normalized)
            }
            pixelValues = concatenated(processedImages)
        }

        // Expand image tokens: 255999 → N copies of 262144
        let startOfImageTokenId = 255999
        let imageTokenId = config.imageTokenId
        let numImageTokens = config.imageSeqLength

        var expandedTokens: [Int32] = []
        for token in promptTokens {
            if token == startOfImageTokenId {
                expandedTokens.append(contentsOf:
                    Array(repeating: Int32(imageTokenId), count: numImageTokens))
            } else {
                expandedTokens.append(Int32(token))
            }
        }

        let tokenArray = MLXArray(expandedTokens).expandedDimensions(axis: 0)
        let mask = MLXArray.ones(like: tokenArray)

        return LMInput(
            text: .init(tokens: tokenArray, mask: mask),
            image: pixelValues.map { .init(pixels: $0) }
        )
    }
}
