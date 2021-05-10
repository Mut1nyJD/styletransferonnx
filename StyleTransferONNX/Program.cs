using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Drawing2D;
using System.Windows.Media.Imaging;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections;
using System.IO;
using System.Diagnostics;

namespace StyleTransferONNX
{
   
  
    class Program
    {

  
        static float[] CreateInputTensorFromImage(String filename,  int[] tensorDims, out float scaleFactor, out int offsetX, out int offsetY)
        {
            var bitmap = new BitmapImage(new Uri(filename));
            var bitmapWidth = bitmap.PixelWidth;
            var bitmapHeight = bitmap.PixelHeight;
            scaleFactor = 1.0f;
            if (tensorDims[3] > 0 && tensorDims[2] > 0)
            {
                if (bitmapWidth > bitmapHeight)
                {
                    scaleFactor = (float)tensorDims[3] / bitmapWidth;
                }
                else
                {
                    scaleFactor = (float)tensorDims[2] / bitmapHeight;
                }
            } else
            {
                tensorDims[3] = bitmapWidth;
                tensorDims[2] = bitmapHeight;
                tensorDims[0] = 1;
                scaleFactor = 1.0f;
            }
            TransformedBitmap tb = new TransformedBitmap(bitmap, new System.Windows.Media.ScaleTransform(scaleFactor, scaleFactor));
            int newWidth = tb.PixelWidth;
            int newHeight = tb.PixelHeight;
            int channels = tb.Format.BitsPerPixel / 8;
            int stride = channels * newWidth;
            byte[] rawData = new byte[stride * newHeight];
            byte[] rawLabelOutput = new byte[tensorDims[2] * tensorDims[3]];
            byte[] rawOutput = new byte[stride * newHeight];
            tb.CopyPixels(rawData, stride, 0);
            int paddingX = tensorDims[3] - newWidth;
            int paddingY = tensorDims[2] - newHeight;
            float[] testData = new float[tensorDims[2] * tensorDims[3] * tensorDims[1]];
            for (int n = 0; n < tensorDims[2] * tensorDims[3] * tensorDims[1]; n++)
                testData[n] = 0.0f;
            offsetX = paddingX / 2;
            offsetY = paddingY / 2;
            // fill up tensor with image data                    
            for (int y = 0; y < newHeight; y++)
            {
                int y1 = y;
                for (int x = 0; x < newWidth; x++)
                {
                    testData[(x + offsetX) + (y + offsetY) * tensorDims[3] + tensorDims[2] * tensorDims[3] * 2] = rawData[(x + y1 * newWidth) * channels];
                    testData[(x + offsetX) + (y + offsetY) * tensorDims[3] + tensorDims[2] * tensorDims[3]] = rawData[(x + y1 * newWidth) * channels + 1];
                    testData[(x + offsetX) + (y + offsetY) * tensorDims[3]] = rawData[(x + y1 * newWidth) * channels + 2];
                }
            }
            return testData;
        }

        static void ProcessOutput(ReadOnlySpan<int> resultDims, float[] tensorData, String outfilename)
        {
            byte[] rawData = new byte[resultDims[2] * resultDims[3] * resultDims[1]];
            for (int y = 0; y < resultDims[2]; y++)
            {
                for (int x = 0; x < resultDims[3]; x++)
                {
                    float red = tensorData[x + y * resultDims[3]];
                    float green = tensorData[x + y * resultDims[3] + resultDims[2] * resultDims[3]];
                    float blue = tensorData[x + y * resultDims[3] + 2 * resultDims[2] * resultDims[3]];
                    if (red < 0.0f)
                        red = 0.0f;
                    if (red > 255.0f)
                        red = 255.0f;
                    if (green < 0.0f)
                        green = 0.0f;
                    if (green > 255.0f)
                        green = 255.0f;
                    if (blue < 0.0f)
                        blue = 0.0f;
                    if (blue > 255.0f)
                        blue = 255.0f;
                    rawData[(x + y * resultDims[3]) * 3] = (byte)red;
                    rawData[(x + y * resultDims[3]) * 3 + 1] = (byte)green;
                    rawData[(x + y * resultDims[3]) * 3 + 2] = (byte)blue;
                }
            }
            var outbitmap = BitmapSource.Create(resultDims[3], resultDims[2], 96, 96,System.Windows.Media.PixelFormats.Rgb24, null, rawData, resultDims[3] * resultDims[1]);
            using (var fileStream = new FileStream(outfilename, FileMode.Create))
            {
                BitmapEncoder encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(outbitmap));
                encoder.Save(fileStream);
            }
        }



        static void Main(string[] args)
        {
            if (args.Length < 3)
            {
                System.Console.WriteLine("Not enough arguments given, use onnx model file input image output image");
                return;
            }
            // when using CPU / MKLDNN provider uncomment the next line
            // var options = new SessionOptions();
            // when using CUDA/GPU Provider if not comment this line
            var options = new SessionOptions();
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.InterOpNumThreads = 8;
            options.IntraOpNumThreads = 4;
            String onnxfile = args[0];
            InferenceSession session = null;
            String inputFilename = args[1];
            String outputFilename = args[2];

            try
            {
                session = new InferenceSession(onnxfile, options);
                var inputMeta = session.InputMetadata;
                int[] inputDim = new int[4];
                float scaleFactor;
                int offsetX, offsetY;
                foreach (var name in inputMeta.Keys)
                {
                    var dim = inputMeta[name].Dimensions;
                    for (int n = 0; n < dim.Length; n++)
                        inputDim[n] = dim[n];
                }
                var testData = CreateInputTensorFromImage(inputFilename, inputDim, out scaleFactor, out offsetX, out offsetY);
                var container = new List<NamedOnnxValue>();

                foreach (var name in inputMeta.Keys)
                {

                    var tensor = new DenseTensor<float>(testData, inputDim);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                using (var results = session.Run(container))
                {
                    int numResults = results.Count;
                    foreach (var r in results)
                    {
                        var resultTensor = r.AsTensor<float>();
                        var resultDimension = resultTensor.Dimensions;
                        var resultArray = resultTensor.ToArray();
                        ProcessOutput(resultDimension, resultArray, outputFilename);
                    }
                }
            }
            catch (Exception e)
            {
                System.Console.WriteLine("Could not load ONNX model, because " + e.ToString());
                return;
            }
            System.Console.WriteLine("Done");
        }
    }
}
