import runpod
from model_loader import get_pipeline
from utils import load_image_input, upload_to_s3, upscale_image, aspect_to_resolution, generate_seed
import logging
import torch

logging.basicConfig(level=logging.INFO)

def handler(job):
   logging.info(f"Received job: {job}")
   if "input" not in job:
	   logging.error("Job missing input field")
	   return {"error": "Job missing input field"}
   inputs = job["input"]
   if "prompt" not in inputs:
	   logging.error("Missing prompt in input")
	   return {"error": "Missing prompt in input"}
   
   image = load_image_input(inputs)
   model = inputs.get("model", "flux_kontext")
   guidance = inputs.get("guidance_scale", 3.5)
   lora_list = inputs.get("lora_list", [])
   steps = inputs.get("steps", 20)
   seed = inputs.get("seed", 0)
   if seed == 0:
      seed = generate_seed()
   aspect_ratio = inputs.get("aspect_ratio")
   upscale_factor = inputs.get("upscale_factor", 1.0)
   pipe = get_pipeline(model, lora_list)
   width, height = aspect_to_resolution(inputs.get("aspect_ratio", "1:1"))
   if model == "flux_dev":
      output = pipe(
	     prompt=inputs["prompt"],
	     width=width,
	     height=height,
	     guidance_scale=guidance,
	     num_inference_steps=steps,
	     generator=torch.Generator("cuda").manual_seed(seed)).images[0]
   else:
      output = pipe(
	     prompt=inputs["prompt"],
	     width=width,
	     height=height,
	     guidance_scale=guidance,
	     num_inference_steps=steps,
	     generator=torch.Generator("cuda").manual_seed(seed)).images[0]	   
   if upscale_factor > 1.0:
	   output = upscale_image(output, upscale_factor)

   res = upload_to_s3(output, job_id=job["id"])
   return {"signedURL": res['url'], "key": res['s3_key'], "Prompt": inputs["prompt"], "Size": str(height) + 'x' + str(width), "Steps": steps, "Seed": seed}

runpod.serverless.start({"handler": handler})
