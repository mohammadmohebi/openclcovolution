/**
 * @author: Mohammad, Vladna
 * @version: 08/2015
 * 
 * */

#include "lodepng.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#include "PACC/Tokenizer.hpp"
#include "TimerC99.hpp"


using namespace std;

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)


void usage(char* inName) {
    cout << endl << "Utilisation> " << inName <<
            " fichier_image fichier_noyau [fichier_sortie=output.png]" << endl;
    exit(1);
}

void decode(const char* inFilename, vector<unsigned char>& outImage,
        unsigned int& outWidth, unsigned int& outHeight) {
    unsigned int lError = lodepng::decode(outImage, outWidth, outHeight, inFilename);

    if (lError)
        cout << "Erreur de dÃ©codage " << lError << ": " <<
            lodepng_error_text(lError) << endl;
}


void encode(const char* inFilename, vector<unsigned char>& inImage,
        unsigned inWidth, unsigned inHeight) {
    unsigned lError = lodepng::encode(inFilename, inImage, inWidth, inHeight);

    if (lError)
        cout << "Erreur d'encodage " << lError << ": " <<
            lodepng_error_text(lError) << endl;
}

int main(int inArgc, char *inArgv[]) {
    cout << "-------------------------------------------" << endl;
    if (inArgc < 3 or inArgc > 4) usage(inArgv[0]);
    string lFilename = inArgv[1];
    string lOutFilename;
    if (inArgc == 4)
        lOutFilename = inArgv[3];
    else
        lOutFilename = "output.png";

    ifstream lConfig;
    lConfig.open(inArgv[2]);
    if (!lConfig.is_open()) {
        cerr << "Le fichier noyau fourni (" << inArgv[2] <<
                ") est invalide." << endl;
        exit(1);
    }

    PACC::Tokenizer lTok(lConfig);
    lTok.setDelimiters(" \n", "");

    string lToken;
    lTok.getNextToken(lToken);

    unsigned int lK = atoi(lToken.c_str());

    cout << "Taille du noyau: " << lK << endl;

    int lFilterSize = lK * lK;
    unsigned int lHalfK = lK / 2;

    double* lFilter = new double[lFilterSize];

    for (unsigned int i = 0; i < lK; i++) {
        for (unsigned int j = 0; j < lK; j++) {
            lTok.getNextToken(lToken);
            lFilter[i * lK + j] = atof(lToken.c_str());
        }
    }

    unsigned int lWidth, lHeight;
    vector<unsigned char> lImageInit;

    decode(lFilename.c_str(), lImageInit, lWidth, lHeight);

    int lInitialImageSize = lImageInit.size();

    const int WORK_GROUP_SIZE = 256;
    cout << "WORK_GROUP_SIZE: " << WORK_GROUP_SIZE << endl;
    
    int lbigHeight = (lHeight / WORK_GROUP_SIZE * WORK_GROUP_SIZE + WORK_GROUP_SIZE);

    vector<unsigned char> lZerosVector((lbigHeight - lHeight) * lWidth * 4);
    
    vector<unsigned char> lImage;
    lImage.reserve(lImageInit.size() + lZerosVector.size()); 
    lImage.insert(lImage.end(), lImageInit.begin(), lImageInit.end());
    lImage.insert(lImage.end(), lZerosVector.begin(), lZerosVector.end());

    int lImageSize = lImage.size();

    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("convolution_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Erreur pendant la lecture du noyau OpenCL.\n");
        exit(1);
    }
    source_str = (char*) malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Obtenir de l'information sur le platform et le device
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1,
            &device_id, &ret_num_devices);

    // Créer le contexte OpenCL
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Créer la queue de commande 
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Créer des buffers de mémoire sur le device pour les vecteur d'entrée, de sortie et
    // du filtre
    cl_mem input_mem_objet = clCreateBuffer(context, CL_MEM_READ_ONLY,
            lImageSize * sizeof (unsigned char), NULL, &ret);
    cl_mem filter_mem_objet = clCreateBuffer(context, CL_MEM_READ_ONLY,
            lFilterSize * sizeof (double), NULL, &ret);
    cl_mem output_mem_objet = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
            lImageSize * sizeof (unsigned char), NULL, &ret);

    // Copier le vecteur d'entrée et le filtre dans les buffers de mémoire correspondants
    ret = clEnqueueWriteBuffer(command_queue, input_mem_objet, CL_TRUE, 0,
            lImageSize * sizeof (unsigned char), &lImage[0], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, filter_mem_objet, CL_TRUE, 0,
            lFilterSize * sizeof (double), lFilter, 0, NULL, NULL);

    // Créer le programme à partir du code source du noyau
    cl_program program = clCreateProgramWithSource(context, 1,
            (const char **) &source_str, (const size_t *) &source_size, &ret);

    // Construire le programme
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Créer le noyau OpenCL
    cl_kernel kernel = clCreateKernel(program, "convolution_kernel", &ret);

    // Définir les arguments du noyau
    ret = clSetKernelArg(kernel, 0, sizeof (cl_mem), (void *) &input_mem_objet);
    ret = clSetKernelArg(kernel, 1, sizeof (cl_mem), (void *) &filter_mem_objet);
    ret = clSetKernelArg(kernel, 2, sizeof (cl_mem), (void *) &output_mem_objet);
    ret = clSetKernelArg(kernel, 3, sizeof (unsigned int), &lHalfK);

    // Démarrer le timer
    TimerC99 timer;

    // Exécuter le noyau OpenCL
    size_t global_item_size[2]; 
    global_item_size[0] = lWidth;
    global_item_size[1] = lbigHeight; 
    size_t local_item_size[2];
    local_item_size[0] = 1;
    local_item_size[1] = WORK_GROUP_SIZE;
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
            global_item_size,
            local_item_size,
            0, NULL, NULL);

    // Lire le buffer de mémoire de sortie dans une variable locale
    unsigned char *lOutputImage = (unsigned char*) malloc(sizeof (unsigned char)*lImageSize);
    ret = clEnqueueReadBuffer(command_queue, output_mem_objet, CL_TRUE, 0,
            lImageSize * sizeof (unsigned char), lOutputImage, 0, NULL, NULL);

    cout << "Temps requis pour le filtrage: " << timer.getElapsedTime() << endl;

    // Créer l'image résultat
    vector<unsigned char> lOutput(lOutputImage, lOutputImage + lImageSize);
    encode(lOutFilename.c_str(), lOutput, lWidth, lHeight);

    lOutput.erase(lOutput.begin() + lInitialImageSize, lOutput.begin() + lImageSize);

    cout << "L'image a été filtrée et enregistrée dans " << lOutFilename <<
            " avec succès!" << endl;

    // Libérer de la mémoire alouée
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(input_mem_objet);
    ret = clReleaseMemObject(filter_mem_objet);
    ret = clReleaseMemObject(output_mem_objet);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(lOutputImage);
    delete lFilter;
    return 0;
}
