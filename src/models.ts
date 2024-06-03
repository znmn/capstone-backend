import * as tf from "@tensorflow/tfjs-node";
import type { L2Args } from "@tensorflow/tfjs-layers/dist/regularizers";
import type { LayersModel } from "@tensorflow/tfjs-layers/dist/engine/training";
import type { SerializableConstructor, Serializable } from "@tensorflow/tfjs-core/dist/serialization";
import path from "path";

interface Prediction {
	plant: string;
	label: string;
	confidence: number | string;
}

class L2 {
	static className = "L2";

	constructor(config: L2Args) {
		return tf.regularizers.l1l2(config);
	}
}

function round(value: number, decimals: number) {
	return Number(Math.round(Number(value + "e" + decimals)) + "e-" + decimals);
}

export class Model {
	private static instance: Model;
	private models: Map<string, LayersModel>;

	constructor() {
		tf.serialization.registerClass(L2 as SerializableConstructor<Serializable>);
		this.models = new Map();
	}

	private static getInstance() {
		if (!Model.instance) {
			Model.instance = new Model();
		}
		return Model.instance;
	}

	private async loadModel(modelPath: string): Promise<LayersModel> {
		modelPath = modelPath.trim().replace(/\\/g, "/");
		const modelName = modelPath.replace(/[^a-zA-Z0-9]/g, "_");

		const isLocal = !modelPath.startsWith("http");
		if (isLocal) {
			modelPath = path.resolve(__dirname, modelPath);
		}

		let model = this.models.get(modelName);
		if (model) {
			return model;
		}

		model = await tf.loadLayersModel(isLocal ? `file://${modelPath}` : modelPath);
		this.models.set(modelName, model);

		return model;
	}

	static async predictImage(imageBuffer: Buffer, modelPath: string, plant: string, labels: string[]): Promise<Prediction> {
		const model = await Model.getInstance().loadModel(modelPath);
		const result = tf.tidy(() => {
			let imgTensor = tf.node.decodeImage(imageBuffer);
			imgTensor = tf.image.resizeBilinear(imgTensor, [150, 150]);
			imgTensor = imgTensor.toFloat().div(255.0);
			imgTensor = imgTensor.expandDims(0);
			const prediction = model.predict(imgTensor) as tf.Tensor;
			const values = prediction.arraySync() as number[];

			return values.flat();
		});

		const max = Math.max(...result);
		const index = result.indexOf(max);
		return { plant, label: labels[index], confidence: round(max, 3) };
	}
}
