import { Hono } from "hono";
import { Model } from "./models";
import fs from "fs";
import path from "path";

const myModelsPath = path.resolve(__dirname, "models.json");
const myModels: {
	id: number;
	name: string;
	model: string;
	labels: string[];
}[] = JSON.parse(fs.readFileSync(myModelsPath, "utf-8"));
const findModel = (plant: string) => {
	return myModels.find((model) => model.name == plant.trim());
};

const predict = new Hono().basePath("/predict");

predict
	.post("/:plant", async (c) => {
		const body = await c.req.parseBody();
		const { plant } = c.req.param();
		const { image } = body;

		const myModel = findModel(plant);
		if (!myModel) {
			return c.json({ message: "Invalid plant" }, 400);
		}

		if (!image) {
			return c.json({ message: "Missing image" }, 400);
		}

		if (!(image instanceof File) || image.type.split("/")[0] !== "image") {
			return c.json({ message: "Invalid image type" }, 400);
		}
		if (image.size > 5 * 1024 * 1024) {
			return c.json({ message: "Image size too large (Max 5MB)" }, 400);
		}

		const { model, labels } = myModel;
		const imgBuffer = Buffer.from(await image.arrayBuffer());

		const result = await Model.predictImage(imgBuffer, model, plant, labels);
		return c.json(result);
	})
	.all((c) => {
		return c.json({ message: "Method not allowed" }, 405);
	});

predict.all("/", (c) => {
	return c.json({ message: "You must specify plant (/predict/:plant)" }, 404);
});

const app = new Hono();

app.get("/", (c) => {
	return c.text("Hello, i'm Zain!");
});

app.route("/", predict);

// export default app
export default {
	port: 3000,
	fetch: app.fetch,
};
