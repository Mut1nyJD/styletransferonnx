// StyleTransferONNX.cpp : Simple C++ program to execute an arbitrary ONNX 
//                         StyleTransfer Network on an image file
//                         using ONNXRuntime and FreeImage
//                         Feel free to use/alterate and incooperate anywhere
//

#include <omp.h>
#include <iostream>
#include<vector>
#include<list>
#include<omp.h>
#include <codecvt>
#include <locale>

// FREEImage library (https://freeimage.sourceforge.io/index.html)
#include "Freeimage.h"
// ONNXRuntime (https://github.com/Microsoft/onnxruntime)
#include "onnxruntime_c_api.h"
// ONNXRuntime DirectML
//#include "dml_provider_factory.h"
// CUDA ONNXRuntime
//#include "cuda_provider_factory.h"


unsigned DLL_CALLCONV
myReadProc(void *buffer, unsigned size, unsigned count, fi_handle handle) {
	return (unsigned)fread(buffer, size, count, (FILE *)handle);
}

unsigned DLL_CALLCONV
myWriteProc(void *buffer, unsigned size, unsigned count, fi_handle handle) {
	return (unsigned)fwrite(buffer, size, count, (FILE *)handle);
}

int DLL_CALLCONV
mySeekProc(fi_handle handle, long offset, int origin) {
	return fseek((FILE *)handle, offset, origin);
}

long DLL_CALLCONV
myTellProc(fi_handle handle) {
	return ftell((FILE *)handle);
}

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);


void CheckStatus(OrtStatus* status)
{
	if (status != NULL) {
		const char* msg = g_ort->GetErrorMessage(status);
		fprintf(stderr, msg);
	}
}



void ProcessInputImage(FILE* fp, std::vector<float>& output, std::vector<int64_t>& inDims, float& scale, __int64& deltaX, __int64& deltaY)
{
	FreeImageIO io;
	io.read_proc = myReadProc;
	io.write_proc = myWriteProc;
	io.seek_proc = mySeekProc;
	io.tell_proc = myTellProc;
	FREE_IMAGE_FORMAT fif = FreeImage_GetFileTypeFromHandle(&io, (fi_handle)fp, 0);
	if (fif != FIF_UNKNOWN)
	{
		FIBITMAP *dib = FreeImage_LoadFromHandle(fif, &io, (fi_handle)fp, 0);
		unsigned int height = FreeImage_GetHeight(dib);
		unsigned int width = FreeImage_GetWidth(dib);
		unsigned int bpp = FreeImage_GetBPP(dib);
		__int64 newWidth = width;
		__int64 newHeight = height;
		if (inDims[2] != -1 && inDims[3] != -1)
		{
			if (height > width)
			{
				newHeight = inDims[2];
				newWidth = (newHeight * width) / height;
			}
			else {
				newWidth = inDims[3];
				newHeight = (newWidth * height) / width;
			}
		}
		else {
			inDims[0] = 1;
			inDims[2] = height;
			inDims[3] = width;
			newWidth = width;
			newHeight = height;
		}
		scale = float(width) / float(newWidth);

		__int64 outWidth = inDims[3];
		__int64 outHeight = inDims[2];
		output.resize(outWidth*outHeight * 3);
		for (int n = 0; n < outWidth * outHeight * 3; n++)
			output[n] = 0.0f;
		deltaX = (outWidth - newWidth) >> 1;
		deltaY = (outHeight - newHeight) >> 1;

#pragma omp parallel for
		for (int y = 0; y < newHeight; y++)
		{
			// Y-Direction is flipped
			int srcY = height - 1 - (int)(y * scale);
			BYTE* bits = FreeImage_GetScanLine(dib, srcY);

			for (int x = 0; x < newWidth; x++)
			{
				int srcX = int(x*scale);
				float red = float(bits[srcX * (bpp == 32 ? 4 : 3) + 2]);
				float green = float(bits[srcX * (bpp == 32 ? 4 : 3) + 1]);
				float blue = float(bits[srcX * (bpp == 32 ? 4 : 3)]);
				output[((x + deltaX) + (y + deltaY) * outWidth)] = red;
				output[((x + deltaX) + (y + deltaY) * outWidth) + outWidth * outHeight] = green;
				output[((x + deltaX) + (y + deltaY) * outWidth) + outWidth * outHeight * 2] = blue;
			}
		}
		FreeImage_Unload(dib);
	}
}


void ProcessOutput(float* tensorData, std::vector<int64_t>& resultDims, std::string& outFilename)
{
	if (resultDims.size() < 4)
		return;
	FIBITMAP* dib = FreeImage_Allocate(resultDims[3], resultDims[2], 24);
	size_t index = outFilename.rfind(".jpg");
	size_t index2 = outFilename.rfind(".JPG");

	FREE_IMAGE_FORMAT outFormat = FIF_PNG;
	if (index != std::string::npos || index2 != std::string::npos)
		outFormat = FIF_JPEG;
	index = outFilename.rfind(".jpeg");
	index2 = outFilename.rfind(".JPEG");
	if (index != std::string::npos || index2 != std::string::npos)
		outFormat = FIF_JPEG;
	index = outFilename.rfind(".bmp");
	index2 = outFilename.rfind(".bmp");
	if (index != std::string::npos || index2 != std::string::npos)
		outFormat = FIF_BMP;


#pragma omp parallel for
	for (int y = 0; y < resultDims[2]; y++)
	{
		BYTE* pBits = FreeImage_GetScanLine(dib, resultDims[2] - 1 - y);
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
			pBits[FI_RGBA_RED] = (unsigned char)red;
			pBits[FI_RGBA_GREEN] = (unsigned char)green;
			pBits[FI_RGBA_BLUE] = (unsigned char)blue;
			pBits += 3;
		}
	}
	FreeImage_Save(outFormat, dib, outFilename.c_str());
	FreeImage_Unload(dib);
}



int main(int argc, char** argv)
{

	if (argc < 4)
	{
		fprintf(stderr, "Not enough arguments provided use onnx_model_file input_image output_image\n");
		return -1;
	}
	std::string outfilename = std::string(argv[3]);
	std::string asciiModelName = std::string(argv[1]);
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
	using convert_t = std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_t, wchar_t> strconvert;
	std::basic_string<wchar_t> modelName = strconvert.from_bytes(asciiModelName);
#endif
	OrtEnv* env = NULL;
	CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "test", &env));
	// initialize session options if needed
	OrtSessionOptions* session_options;
	CheckStatus(g_ort->CreateSessionOptions(&session_options));
//	OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0); FOR CUDA
//	OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0); FOR DIRECTML
	g_ort->SetIntraOpNumThreads(session_options, 4);
	g_ort->SetInterOpNumThreads(session_options, 8);
	OrtSession* session = NULL;
	OrtSession* session_seg = NULL;
#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
	CheckStatus(g_ort->CreateSession(env, modelName.c_str(), session_options, &session));
#else
	CheckStatus(g_ort->CreateSession(env, asciiModelName.c_str(), session_options, &session));
#endif
	size_t num_input_nodes;
	OrtStatus* status = NULL;
	OrtAllocator* allocator = NULL;
	CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));
	CheckStatus(g_ort->SessionGetInputCount(session, &num_input_nodes));
	std::vector<const char*> input_node_names(num_input_nodes);
	std::vector<int64_t> input_node_dims;

	fprintf(stderr, "Number of inputs = %zu\n", num_input_nodes);
	for (size_t i = 0; i < num_input_nodes; i++) {
		// print input node names
		char* input_name;
		CheckStatus(g_ort->SessionGetInputName(session, i, allocator, &input_name));
		printf("Input %zu : name=%s\n", i, input_name);
		input_node_names[i] = input_name;
		// print input node types
		OrtTypeInfo* typeinfo;
		CheckStatus(g_ort->SessionGetInputTypeInfo(session, i, &typeinfo));
		const OrtTensorTypeAndShapeInfo* tensor_info;
		g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
		ONNXTensorElementDataType type;
		g_ort->GetTensorElementType(tensor_info, &type);
		fprintf(stdout, "Input %zu : type=%d\n", i, type);
		// print input shapes/dims
		size_t num_dims;
		g_ort->GetDimensionsCount(tensor_info, &num_dims);
		fprintf(stdout, "Input %zu : num_dims=%zu\n", i, num_dims);
		input_node_dims.resize(num_dims);
		g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims);
		for (size_t j = 0; j < num_dims; j++)
			fprintf(stdout, "Input %zu : dim %zu=%jd\n", i, j, input_node_dims[j]);
		g_ort->ReleaseTypeInfo(typeinfo);

	}
	OrtMemoryInfo* memory_info = NULL;
	CheckStatus(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
	std::vector<float> input_tensor_values;
	FILE* fp = NULL;
	fopen_s(&fp, argv[2], "rb");
	if (fp != NULL)
	{
		float scale;
		__int64 xOffset, yOffset;
		ProcessInputImage(fp, input_tensor_values, input_node_dims, scale, xOffset, yOffset);
		fclose(fp);
		OrtValue* input_tensor = NULL;
		OrtValue* output_tensors[1] = { NULL };

		size_t tensor_size = input_tensor_values.size();
		CheckStatus(g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), tensor_size * sizeof(float), input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

		int is_tensor;
		g_ort->IsTensor(input_tensor, &is_tensor);
		if (is_tensor)
		{
			size_t numOut;
			g_ort->SessionGetOutputCount(session, &numOut);
			std::vector<char*> output_names;
			std::vector<std::vector<int64_t> > output_nodes_dims;
			std::vector<int64_t>  output_node_dims;
			for (size_t i = 0; i < numOut; i++)
			{
				char* output_name;
				g_ort->SessionGetOutputName(session, i, allocator, &output_name);
				output_names.push_back(output_name);
				printf("OutputName(%d)=%s\n", int(i), output_name);
			}
			g_ort->Run(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_names.data(), output_names.size(), output_tensors);
			float *dataOut = NULL;
			g_ort->IsTensor(output_tensors[0], &is_tensor);
			if (is_tensor)
			{
				OrtTensorTypeAndShapeInfo* tensorTandS;
				size_t dimOut;
				CheckStatus(g_ort->GetTensorTypeAndShape(output_tensors[0], &tensorTandS));
				CheckStatus(g_ort->GetDimensionsCount(tensorTandS, &dimOut));
				output_node_dims.resize(dimOut);
				g_ort->GetDimensions(tensorTandS, (int64_t*)output_node_dims.data(), dimOut);
				CheckStatus(g_ort->GetTensorMutableData(output_tensors[0], (void**)&dataOut));
				ProcessOutput(dataOut, output_node_dims,outfilename);
			}
			g_ort->ReleaseValue(output_tensors[0]);
			g_ort->ReleaseValue(input_tensor);
			input_tensor = NULL;
			output_tensors[0] = NULL;
		}
	}
	return 0;
}

