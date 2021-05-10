import argparse
from PIL import Image
import numpy as np
import onnxruntime as rt

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="StyleTransferONNX")
   parser.add_argument('--model', type=str, default=' ', help='ONNX model file', required=True)
   parser.add_argument('--input', type=str, default=' ', help='Input image', required=True)
   parser.add_argument('--output', type=str, default=' ', help='learning rate',required=True)
   args = parser.parse_args()
   session = rt.InferenceSession(args.model)
   inputH = session.get_inputs()
   outputH = session.get_outputs()
   img = Image.open(args.input)
   print('img dim: ',img.width,' ',img.height)
   inputArray = np.asarray(img)
   inputArray = inputArray.astype(np.float32);
   inputArray = inputArray.transpose([2,0,1])
   np.clip(inputArray,0,255,out=inputArray)
   inputArray = inputArray.reshape((1,3,img.height,img.width))
   output_res = session.run(None,{inputH[0].name: inputArray})
   output_img =  output_res[0].reshape(3,output_res[0].shape[2],output_res[0].shape[3])
   output_img = output_img.transpose([1,2,0])
   output_img = output_img.astype(np.uint8)
   output = Image.fromarray(output_img)
   output.save(args.output)
