/* --------------------------------- CNN.h ---------------------------------
 *
 * A single header implementation of Convolution Neural Networks.
 *
 * #define CNN_IMPLEMENTATION before including the header in 
 * exactly one translation unit.
 * 
 * 
 * Important Macros -
 *    - CNN_EXIT_ON_ERROR: 
 *          Exits when an error is encountered use only during development,
 *          It may leak memory.
 *    - CNN_ASSERT:
 *          USE ONLY WHEN IT'S UR LAST HOPE GOODLUCK!
 *          Uses assert() when error occurs, helps when debugging. 
 *          Shows which line raised the error.
 * 
 * Redefinable Macros -
 *    - CNNDEF: 
 *          Defaults to external linkage
 *          Override CNNDEF to `static inline` when using CNN.h as 
 *          header only and want the compiler to remove unused functions.
 * 
 * Author : px7nn
 * GitHub : https://github.com/px7nn/CNN.h
 * License: MIT 
 * 
 * --------------------------------- ----- --------------------------------- */


// Development Notes:
//    This library is currently under developement. Hence frequent changes 
//    in API is expected.
//    Once the library ready for a stable release all layman implementations
//    will be replaced (e.g overflow).


#ifndef CNN_H
#define CNN_H

#define CNN_VERSION_MAJOR   0
#define CNN_VERSION_MINOR   1
#define CNN_VERSION_PATH    0
#define CNN_VERSION_STRING  "0.1.0"

#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>

#ifndef CNNDEF
    #define CNNDEF
#endif  // CNNDEF

#ifdef CNN_ASSERT
    #include <assert.h>
    #ifdef CNN_IMPLEMENTATION
        #define _cnn_raise_error(err) do {      \
            _cnn_last_err = (CNN_Error) err;    \
            assert((CNN_Error) err == CNN_OK);  \
        } while(0)
    #else
        #define _cnn_raise_error(err) do {      \
            assert((CNN_Error) err == CNN_OK);  \
        } while(0)
    #endif
#endif

#define TENSOR_AT(TENS, ch, i, j) ((TENS)->data[(ch)][(TENS)->width * (i) + (j)])

#define CL_W_AT(cl, oc, ic, ki, kj)           \
    ((cl)->weights[                           \
        (((oc) * (cl)->input_channels + (ic)) \
            * (cl)->kernel_height + (ki))     \
            * (cl)->kernel_width + (kj)       \
    ])

#define DL_W_AT(dl, i, j) ((dl)->weights[(i) * (dl)->input_size + (j)])

typedef enum {
    CNN_OK,                 // No Error

    CNN_ERR_ALLOC_FAILED,   // Heap M/M Alloc failed
    CNN_ERR_INVALID_ARG,    // An argument was structurally invalid (e.g. zero-sized)
    
    CNN_ERR_INVALID_STATE,  // Structure is not properly initialized
    CNN_ERR_SHAPE_MISMATCH, // Incompatible dimensions
    CNN_ERR_COUNT
} CNN_Error;

typedef size_t      s_t;
typedef ptrdiff_t   p_t;

typedef struct {
    s_t width, height;
    s_t channels;
    float **data;
} CNN_Tensor;

typedef struct {
    s_t input_channels;
    s_t output_channels;

    s_t stride, padding;

    s_t kernel_width;
    s_t kernel_height;

    float *weights, *bias;
    float *d_weights, *d_bias;

    CNN_Tensor *output;
} CNN_ConvLayer;

typedef struct {
    s_t input_size;
    s_t output_size;

    float *weights; // out * in
    float *bias;    // out
    float *output;  // out

    float *d_weights;
    float *d_bias;
} CNN_DenseLayer;

CNNDEF CNN_Tensor *CNN_Create_Tensor(s_t width, s_t height, s_t channels);
CNNDEF void        CNN_Display_Tensor(const CNN_Tensor *tensor);
CNNDEF void        CNN_Destroy_Tensor(CNN_Tensor *tensor);

CNNDEF CNN_ConvLayer *CNN_Create_ConvLayer(
    s_t input_channels,
    s_t output_channels,
    s_t stride,
    s_t padding,
    s_t kernel_width,
    s_t kernel_height
);
CNNDEF void CNN_Display_ConvLayer(const CNN_ConvLayer *cl);
CNNDEF void CNN_Display_Weights(const CNN_ConvLayer *cl);
CNNDEF void CNN_Destroy_ConvLayer(CNN_ConvLayer *cl);

CNNDEF bool CNN_Conv_Output_Shape(
    const CNN_Tensor *input, 
    const CNN_ConvLayer *cl,
    s_t *out_width,
    s_t *out_height
);

CNNDEF bool CNN_Conv_Forward(
    const CNN_Tensor *input,
    CNN_ConvLayer *cl
);
CNNDEF bool CNN_Conv_Backward(
    const CNN_Tensor *input,
    CNN_ConvLayer *cl,
    const float *d_output,
    s_t d_output_length,
    float *d_input
);
CNNDEF bool CNN_Conv_Update(
    CNN_ConvLayer *cl,
    float learning_rate
);

CNNDEF bool CNN_ReLU(CNN_Tensor *tensor);

CNNDEF float *CNN_Flatten(
    const CNN_Tensor *tensor,
    s_t *out_length,
    float *buf
);

CNNDEF CNN_DenseLayer *CNN_Create_DenseLayer(s_t input_size, s_t output_size);
CNNDEF void CNN_Destroy_DenseLayer(CNN_DenseLayer *dl);
CNNDEF bool CNN_Dense_Forward(const float *input, s_t input_length, CNN_DenseLayer *dl);
CNNDEF bool CNN_Dense_Backward(
    CNN_DenseLayer *dl,
    const float *input,
    const float *probabilities,
    s_t target,
    float *d_input  // NULLABLE
);
CNNDEF bool CNN_Dense_Update(
    CNN_DenseLayer *dl,
    float learning_rate
);

CNNDEF bool CNN_Softmax(const float *input, s_t input_length, float *output);
CNNDEF bool CNN_CrossEntropy(const float *probabilites, s_t length, s_t target, float *out_loss);

CNNDEF bool CNN_TRAIN(
    CNN_ConvLayer  *conv,
    CNN_DenseLayer *dense,
    const CNN_Tensor **images,
    const s_t *labels,
    s_t sample_count,
    s_t epochs,
    float conv_learning_rate,
    float dense_learning_rate,
    float *epoch_losses
);
CNNDEF void CNN_Perror(const char *str);

#ifndef CNN_ASSERT
    static void        _cnn_raise_error(CNN_Error err);
#endif

static const char *_cnn_err_str(CNN_Error err);
static void        _cnn_exit_on_err();
static bool        _cnn_valid_s_t(s_t argc, ...);
static bool        _cnn_valid_tensor(const CNN_Tensor *t);
static bool        _cnn_valid_cl(const CNN_ConvLayer *cl);
static bool        _cnn_valid_dl(const CNN_DenseLayer *dl);

static float _cnn_math_sqrt(float x);
static float _cnn_math_expf(float x);
static float _cnn_math_logf(float x);
static float _cnn_init_weight(s_t input_count);

// -----------------------------------------------------------------------------

#ifdef CNN_IMPLEMENTATION

static CNN_Error _cnn_last_err = CNN_OK; 

#ifndef CNN_ASSERT
    static void _cnn_raise_error(CNN_Error err){
        _cnn_last_err = err;
        #ifdef CNN_EXIT_ON_ERROR
            _cnn_exit_on_err();
        #endif
    }
#endif
static const char *_cnn_err_str(CNN_Error err){
    switch(err){
        case CNN_OK:                return "No error";
        case CNN_ERR_ALLOC_FAILED:  return "Memory allocation failed";
        case CNN_ERR_INVALID_ARG:   return "An argument was invalid";
        case CNN_ERR_INVALID_STATE: return "A given structure was not properly initialized";
        case CNN_ERR_SHAPE_MISMATCH:return "Tensor shape mismatch";

        default:                    return "Unknown Error";
    }
}

static void _cnn_exit_on_err(){
    CNN_Perror("CNN_EXIT_ON_ERROR");
    exit(EXIT_FAILURE);
}

static bool _cnn_valid_s_t(s_t argc, ...){
    va_list args;
    va_start(args, argc);
    for(s_t i=0; i<argc; ++i)
        if(va_arg(args, s_t) == 0){
            va_end(args);
            return false;
        }
    va_end(args);
    return true;
}

static bool _cnn_valid_tensor(const CNN_Tensor *t){
    if(!t || !t->data || !_cnn_valid_s_t(3,
        t->width, t->height, t->channels))
        return false;
    for(s_t i=0; i<t->channels; ++i)
        if(!t->data[i]) return false;
    return true;
}

static bool _cnn_valid_cl(const CNN_ConvLayer *cl){
    if(!cl || !cl->bias || !cl->weights || 
        !_cnn_valid_s_t(5, cl->kernel_width, 
        cl->kernel_height, cl->stride,
        cl->input_channels, cl->output_channels))
        return false;
    return true;
}

static bool _cnn_valid_dl(const CNN_DenseLayer *dl){
    if(!dl || !dl->weights || !dl->bias || !dl->output || 
        !dl->input_size || !dl->output_size)
        return false;
    return true;
}

static float _cnn_math_sqrt(float x){
    if(x <= 0.f) return 0.f;
    float y = x;
    for(s_t i=0; i<5; ++i)
        y = 0.5f * (y + x/y);
    return y;
}

static float _cnn_math_expf(float x){
    if(x != x) // NaN
        return x;
    if(x > 88.0f)
        return 3.402823466e38f; // approx. FLT_MAX
    if(x < -103.f)
        return 0.f;
    float r = x/16.f;
    float term = 1.f, sum = 1.f;
    
    for(s_t i=1; i<=16; ++i){
        term *= r/(float)i;
        sum += term;
    }

    sum *= sum;
    sum *= sum;
    sum *= sum;
    sum *= sum;
    
    return sum;
}

static float _cnn_math_logf(float x){
    if(x <= 0)
        return -3.402823466e+38f;
    float y = 0.f;
    for(s_t i=0; i<6; ++i){
        float ey = _cnn_math_expf(y);
        y -= (ey - x) / ey;
    }
    return y;
}

static float _cnn_init_weight(s_t input_count){
    float w = ((float)rand() / RAND_MAX) * 2.f - 1.f;
    return _cnn_math_sqrt(6.f / (float)input_count) * w;
}

CNNDEF void CNN_Perror(const char *str){
    fprintf(stderr, "%s: %s\n", str, _cnn_err_str(_cnn_last_err));
}

CNNDEF CNN_Tensor *CNN_Create_Tensor(s_t width, s_t height, s_t channels){
    if(!_cnn_valid_s_t(3, width, height, channels)){
        _cnn_raise_error(CNN_ERR_INVALID_ARG);
        return NULL;
    }
    CNN_Tensor *tensor = malloc(sizeof *tensor);
    if(!tensor)
        goto err;

    tensor->data = malloc(channels * sizeof *tensor->data);
    if(!tensor->data)
        goto err;
        
    for(s_t ch=0; ch<channels; ++ch){
        // LAYMAN: Multiplication overflow
        tensor->data[ch] = calloc(width * height, sizeof **tensor->data);
        
        if(!tensor->data[ch])
            goto err;
    }
    
    tensor->width = width; tensor->height = height;
    tensor->channels = channels;

    return tensor;

    err:
        CNN_Destroy_Tensor(tensor);
        _cnn_raise_error(CNN_ERR_ALLOC_FAILED);
        return NULL;
}

CNNDEF void CNN_Display_Tensor(const CNN_Tensor *tensor){
    if(!tensor){
        printf("CNN_Tensor: NULL\n");
        return;
    }
    if(!_cnn_valid_tensor(tensor)){
        printf("CNN_Tensor: invalid structure\n");
        return;
    }
    printf(
        "+------------- CNN Tensor -------------+\n"
        "| Shape: (C=%-3zu, H=%-3zu, W=%-3zu)         |\n"
        "| Type : float                         |\n",
        tensor->channels, tensor->height, tensor->width
    );
    for(s_t ch = 0; ch < tensor->channels; ++ch){
        printf("|%-38s|\n", " ");
        printf("| Channel %-28zu |\n", ch);
        for(s_t i = 0; i < tensor->height; ++i){
            printf("|    [    ");
            for(s_t j = 0; j < tensor->width; ++j)
                printf("%-8.2f", TENSOR_AT(tensor, ch, i, j));
            printf("]    |\n");
        }
    }
    printf("+--------------------------------------+\n");
}

CNNDEF void CNN_Destroy_Tensor(CNN_Tensor *tensor){
    if(!tensor) return;
    if(!tensor->data){
        free(tensor);
        return;
    }

    for(s_t ch=0; ch<tensor->channels; ++ch)
        if(tensor->data[ch]) free(tensor->data[ch]);

    free(tensor->data);
    free(tensor);
}

CNNDEF CNN_ConvLayer *CNN_Create_ConvLayer(
    s_t input_channels,
    s_t output_channels,
    s_t stride,
    s_t padding,
    s_t kernel_width,
    s_t kernel_height
){
    if(!_cnn_valid_s_t(5,
        input_channels, output_channels, stride,
        kernel_width, kernel_height
    )){
        _cnn_raise_error(CNN_ERR_INVALID_ARG);
        return NULL;
    }
    
    CNN_ConvLayer *cl = calloc(1, sizeof *cl);
    if(!cl){
        _cnn_raise_error(CNN_ERR_ALLOC_FAILED);
        return NULL;
    }

    // LAYMAN: Multiplication overflow
    s_t weight_count = output_channels * input_channels 
                    * kernel_height * kernel_width;

    *cl = (CNN_ConvLayer){
        .input_channels  = input_channels, 
        .output_channels = output_channels, 
        .stride          = stride,
        .padding         = padding, 
        .kernel_width    = kernel_width, 
        .kernel_height   = kernel_height,

        .weights = calloc(
            weight_count, sizeof *cl->weights
        ),
        .bias    = calloc(
            output_channels, sizeof *cl->bias
        ),

        .d_weights = calloc(
            weight_count, sizeof *cl->d_weights
        ),
        .d_bias    = calloc(
            output_channels, sizeof *cl->d_bias
        ),
        .output = NULL
    };

    if(!cl->weights || !cl->bias || !cl->d_weights || !cl->d_bias){
        _cnn_raise_error(CNN_ERR_ALLOC_FAILED);
        CNN_Destroy_ConvLayer(cl);
        return NULL;
    }
    // HE Inititalize weight
    for(s_t oc = 0; oc < cl->output_channels; ++oc)
        for(s_t ic = 0; ic < cl->input_channels; ++ic)
            for(s_t ki=0; ki < cl->kernel_height; ++ki)
                for(s_t kj = 0; kj < cl->kernel_width; ++kj)
                    CL_W_AT(cl, oc, ic, ki, kj) = _cnn_init_weight(weight_count/output_channels);
    return cl;
}

CNNDEF void CNN_Display_ConvLayer(const CNN_ConvLayer *cl){
    if(!cl){
        printf("CNN_ConvLayer: NULL\n");
        return;
    }
    if(!_cnn_valid_cl(cl)){
        printf("CNN_ConvLayer: invalid structure\n");
        return;
    }
    s_t weight_count = cl->output_channels * cl->input_channels
                        * cl->kernel_height * cl->kernel_width;
    s_t bias_count   = cl->output_channels;
    printf(
        "+---------- CNN ConvLayer ----------+\n"
        "| Input channels : %-16zu |\n"
        "| Output channels: %-16zu |\n"
        "| Kernel         : %zu x %-12zu |\n"
        "| Stride         : %-16zu |\n"
        "| Padding        : %-16zu |\n"
        "| Weights        : %-16zu |\n"
        "| Biases         : %-16zu |\n"
        "| Parameters     : %-16zu |\n",
        cl->input_channels,
        cl->output_channels,
        cl->kernel_height,
        cl->kernel_width,
        cl->stride,
        cl->padding,
        weight_count,
        bias_count,
        weight_count + bias_count
    );

    if(!cl->output)
        printf("| Output         : Not computed yet |\n");
    else
        printf(
            "| Output shape   : %-3zu x %-3zu x %-3zu  |\n",
            cl->output->channels,
            cl->output->height,
            cl->output->width
        );
    printf("+-----------------------------------+\n");
}

CNNDEF void CNN_Display_Weights(const CNN_ConvLayer *cl){
    if(!cl){
        printf("CNN_Weights: NULL layer\n");
        return;
    }
    if(!_cnn_valid_cl(cl)){
        printf("CNN_Weights: invalid layer structure\n");
        return;
    }
    printf(
        "+----------- CNN Weights -----------+\n"
        "| Input channels : %-17zu|\n"
        "| Output channels: %-17zu|\n"
        "| Kernel shape   : %3zu x %-3zu        |\n",
        cl->input_channels,
        cl->output_channels,
        cl->kernel_height,
        cl->kernel_width
    );

    for(s_t oc = 0; oc < cl->output_channels; ++oc){
        printf("|%35s|\n", " ");
        printf("| Filter %-27zu|\n", oc);
        printf("| Bias: %-27.4f |\n", cl->bias[oc]);
        for(s_t ic = 0; ic < cl->input_channels; ++ic){
            printf("| Input channel %-20zu|\n", ic);
            for(s_t ky = 0; ky < cl->kernel_height; ++ky){
                printf("|   [   ");
                for(s_t kx = 0; kx < cl->kernel_width; ++kx)
                    printf("%-9.4f", CL_W_AT(cl, oc, ic, ky, kx));
                printf(" ]        |\n");
            }
        }
    }
    printf("+-----------------------------------+\n");
}

CNNDEF void CNN_Destroy_ConvLayer(CNN_ConvLayer *cl){
    if(!cl) return;
    free(cl->weights);
    free(cl->bias);
    free(cl->d_weights);
    free(cl->d_bias);
    CNN_Destroy_Tensor(cl->output);
    free(cl);
}

CNNDEF bool CNN_Conv_Output_Shape(
    const CNN_Tensor *input, 
    const CNN_ConvLayer *cl,
    s_t *out_width,
    s_t *out_height
){
    if(!input || !cl){
        _cnn_raise_error(CNN_ERR_INVALID_ARG);
        return false;
    }
    if(!_cnn_valid_tensor(input) || !_cnn_valid_cl(cl)){ 
        _cnn_raise_error(CNN_ERR_INVALID_STATE);
        return false;
    }
    if(input->channels != cl->input_channels){
        _cnn_raise_error(CNN_ERR_SHAPE_MISMATCH);
        return false;
    }

    s_t padded_width  = input->width  + 2 * cl->padding;
    s_t padded_height = input->height + 2 * cl->padding;

    if(cl->kernel_width > padded_width || 
        cl->kernel_height > padded_height){
            _cnn_raise_error(CNN_ERR_SHAPE_MISMATCH);
            return false;
        }
    
        *out_width  = (padded_width  - cl->kernel_width)  / cl->stride + 1;
        *out_height = (padded_height - cl->kernel_height) / cl->stride + 1;

        return true;
}

CNNDEF bool CNN_Conv_Forward(const CNN_Tensor *input, CNN_ConvLayer *cl){
    s_t out_width, out_height;
    if(!CNN_Conv_Output_Shape(input, cl, &out_width, &out_height))
        return false;
    
    CNN_Destroy_Tensor(cl->output);
    
    cl->output = CNN_Create_Tensor(out_width, out_height, cl->output_channels);
    if(!cl->output)
        return false;
    
    for(s_t oc = 0; oc < cl->output_channels; ++oc){
        for(s_t oi = 0; oi < out_height; ++oi){
            for(s_t oj = 0; oj < out_width; ++oj){

                float sum = cl->bias[oc];

                for(s_t ic = 0; ic < cl->input_channels; ++ic){
                    for(s_t ki = 0; ki < cl->kernel_height; ++ki){
                        for(s_t kj = 0; kj < cl->kernel_width; ++kj){
                            p_t i_i = (p_t)(oi * cl->stride + ki) - (p_t)cl->padding;
                            p_t i_j = (p_t)(oj * cl->stride + kj) - (p_t)cl->padding;
                            if(i_i  < 0 || i_i >= (p_t)input->height ||
                                i_j < 0 || i_j >= (p_t)input->width)
                                continue;

                            float pixel  = TENSOR_AT(input, ic, (s_t)i_i, (s_t)i_j);
                            float weight = CL_W_AT(cl, oc, ic, ki, kj);

                            sum += pixel * weight;
                        }
                    }
                }
                TENSOR_AT(cl->output, oc, oi, oj) = sum;
            }
        }
    }
    return true;
}

CNNDEF bool CNN_Conv_Backward(
    const CNN_Tensor *input,
    CNN_ConvLayer *cl,
    const float *d_output,
    s_t d_output_length,
    float *d_input
){
    if(!input || !cl || !d_output){
        _cnn_raise_error(CNN_ERR_INVALID_ARG);
        return false;
    }
    if(!_cnn_valid_tensor(input) || !_cnn_valid_cl(cl) ||
        !cl->output || !cl->d_weights || !cl->d_bias ||
        !_cnn_valid_tensor(cl->output)){
        _cnn_raise_error(CNN_ERR_INVALID_STATE);
        return false;
    }
    if(input->channels != cl->input_channels){
        _cnn_raise_error(CNN_ERR_SHAPE_MISMATCH);
        return false;
    }
    const s_t out_h = cl->output->height;
    const s_t out_w = cl->output->width;
    const s_t out_c = cl->output_channels;
    if(d_output_length != out_c * out_h * out_w){
        _cnn_raise_error(CNN_ERR_SHAPE_MISMATCH);
        return false;
    }
    s_t weight_count = cl->output_channels * cl->input_channels 
                    * cl->kernel_height * cl->kernel_width;
    for(s_t i=0; i<weight_count; ++i)
        cl->d_weights[i] = 0.f;
    for(s_t i=0; i<cl->output_channels; ++i)
        cl->d_bias[i] = 0.f;
    
    if(d_input){
        s_t input_count = input->channels * input->height * input->width;
        for(s_t i=0; i<input_count; ++i)
            d_input[i] = 0.f;
    }

    for(s_t oc = 0; oc < out_c; ++oc)
        for(s_t oi = 0; oi < out_h; ++oi)
            for(s_t oj = 0; oj < out_w; ++oj){

                s_t out_idx = oc * out_h * out_w + oi * out_w + oj;

                float dz = d_output[out_idx];

                if(cl->output->data[oc][oi * out_w + oj] <= 0.f)
                    dz = 0.f;

                cl->d_bias[oc] += dz;

                for(s_t ic = 0; ic < cl->input_channels; ++ic)
                    for(s_t ki = 0; ki < cl->kernel_height; ++ki)
                        for(s_t kj = 0; kj < cl->kernel_width; ++kj){
                            p_t ii = (p_t)(oi * cl->stride + ki) - (p_t)cl->padding;
                            p_t ij = (p_t)(oj * cl->stride + kj) - (p_t)cl->padding;

                            if(ii < 0 || ij < 0 || ii >= (p_t)input->height || ij >= (p_t)input->width)
                                continue;
                            
                            float x = TENSOR_AT(input, ic, (s_t)ii, (s_t)ij);

                            cl->d_weights[
                                (((oc * cl->input_channels + ic)
                                  * cl->kernel_height + ki)
                                  * cl->kernel_width + kj)
                            ] += x * dz;

                            if(d_input)
                                d_input[
                                    (ic * input->height * input->width +
                                    (s_t)ii * input->width +
                                    (s_t)ij)
                                ] += CL_W_AT(cl, oc, ic, ki, kj) * dz;
                        }
            }
    return true;
}

CNNDEF bool CNN_Conv_Update(CNN_ConvLayer *cl, float learning_rate){
    if(!cl || learning_rate <= 0.f){
        _cnn_raise_error(CNN_ERR_INVALID_ARG);
        return false;
    }
    if(!_cnn_valid_cl(cl) || !cl->d_weights || !cl->d_bias){
        _cnn_raise_error(CNN_ERR_INVALID_STATE);
        return false;
    }

    s_t weight_count = cl->output_channels * cl->input_channels 
                    * cl->kernel_height * cl->kernel_width;
    
    for(s_t i=0; i<weight_count; ++i)
        cl->weights[i] -= learning_rate * cl->d_weights[i];
    for(s_t i=0; i<cl->output_channels; ++i)
        cl->bias[i] -= learning_rate * cl->d_bias[i];
    return true;
}


CNNDEF bool CNN_ReLU(CNN_Tensor *tensor){
    if(!tensor){
        _cnn_raise_error(CNN_ERR_INVALID_ARG);
        return false;
    }
    if(!_cnn_valid_tensor(tensor)){
        _cnn_raise_error(CNN_ERR_INVALID_STATE);
        return false;
    }
    for(s_t ch = 0; ch < tensor->channels; ++ch)
        for(s_t i = 0; i < tensor->height; ++i)
            for(s_t j = 0; j < tensor->width; ++j){
                float x = TENSOR_AT(tensor, ch, i, j);
                TENSOR_AT(tensor, ch, i, j) = (x > 0.f)? x : 0.f;
            }
    return true;
}

CNNDEF float *CNN_Flatten(const CNN_Tensor *tensor, s_t *out_length, float *buf){
    if(!tensor || !out_length || !buf){
        _cnn_raise_error(CNN_ERR_INVALID_ARG);
        return NULL;
    }
    if(!_cnn_valid_tensor(tensor)){
        _cnn_raise_error(CNN_ERR_INVALID_STATE);
        return NULL;
    }
    
    s_t count = tensor->channels * tensor->height * tensor->width;
    // float *flat = malloc(count * sizeof *flat);
    // if(!flat){
    //     _cnn_raise_error(CNN_ERR_ALLOC_FAILED);
    //     return NULL;
    // }

    s_t idx = 0;
    for(s_t ch = 0; ch < tensor->channels; ++ch)
        for(s_t i = 0; i < tensor->height; ++i)
            for(s_t j = 0; j < tensor->width; ++j)
                buf[idx++] = TENSOR_AT(tensor, ch, i, j);
    *out_length = count;
    // return flat;
}

CNNDEF CNN_DenseLayer *CNN_Create_DenseLayer(s_t input_size, s_t output_size){
    if(!input_size || !output_size){
        _cnn_raise_error(CNN_ERR_INVALID_ARG);
        return NULL;
    }
    
    CNN_DenseLayer *dl = calloc(1, sizeof *dl);
    if(!dl) goto err;
    
    *dl = (CNN_DenseLayer){
        .input_size = input_size,
        .output_size = output_size,
        .weights = malloc(
            input_size * output_size * sizeof *dl->weights
        ),
        .bias = calloc(
            output_size, sizeof *dl->bias
        ),
        .output = calloc(
            output_size, sizeof *dl->output
        ),
        .d_weights = calloc(
            input_size * output_size, sizeof *dl->d_weights
        ),
        .d_bias = calloc(
            output_size, sizeof *dl->d_bias
        )
    };

    if(!dl->weights || !dl->bias || !dl->output)
        goto err;

    for(s_t i=0; i<output_size; ++i)
        for(s_t j=0; j<input_size; ++j)
            DL_W_AT(dl, i, j) = _cnn_init_weight(input_size);
    
    return dl;

    err:
        CNN_Destroy_DenseLayer(dl);
        _cnn_raise_error(CNN_ERR_ALLOC_FAILED);
        return NULL;
}

CNNDEF void CNN_Destroy_DenseLayer(CNN_DenseLayer *dl){
    if(!dl) return;
    free(dl->weights);
    free(dl->bias);
    free(dl->output);
    free(dl->d_weights);
    free(dl->d_bias);
    free(dl);
}

CNNDEF bool CNN_Dense_Forward(const float *input, s_t input_length, CNN_DenseLayer *dl){
    if(!input || !input_length || !dl){
        _cnn_raise_error(CNN_ERR_INVALID_ARG);
        return false;
    }
    if(!_cnn_valid_dl(dl)){
        _cnn_raise_error(CNN_ERR_INVALID_STATE);
        return false;
    }
    if(input_length != dl->input_size){
        _cnn_raise_error(CNN_ERR_SHAPE_MISMATCH);
        return false;
    }
    for(s_t i=0; i<dl->output_size; ++i){
        float sum = dl->bias[i];
        for(s_t j=0; j<dl->input_size; ++j)
            sum += DL_W_AT(dl, i, j) * input[j];
        dl->output[i] = sum;
    }
    return true;
}

CNNDEF bool CNN_Softmax(const float *input, s_t input_length, float *output){
    if(!input || !input_length || !output){
        _cnn_raise_error(CNN_ERR_INVALID_ARG);
        return false;
    }
    float max = input[0];
    for(s_t i=1; i<input_length; ++i)
        if(max < input[i])
            max = input[i];
    
    double sum = 0;
    for(s_t i=0; i<input_length; ++i){
        output[i] = _cnn_math_expf(input[i] - max);
        sum += output[i];
    }

    for(s_t i=0; i<input_length; ++i)
        output[i] /= sum;

    return true;
}

CNNDEF bool CNN_CrossEntropy(const float *probabilites, s_t length, s_t target, float *out_loss){
    if(!probabilites || !out_loss || !length || target >= length){
        _cnn_raise_error(CNN_ERR_INVALID_ARG);
        return false;
    }

    float p = probabilites[target];
    const float EPS = 1e-7f;
    if(p < EPS) p = EPS;

    *out_loss = -_cnn_math_logf(p);
    return true;
}

CNNDEF bool CNN_Dense_Backward(
    CNN_DenseLayer *dl,
    const float *input,
    const float *probabilities,
    s_t target,
    float *d_input
){
    if(!dl || !input || !probabilities || target >= dl->output_size){
        _cnn_raise_error(CNN_ERR_INVALID_ARG);
        return false;
    }
    if(!_cnn_valid_dl(dl) || !dl->d_weights || !dl->d_bias){
        _cnn_raise_error(CNN_ERR_INVALID_STATE);
        return false;
    }

    if(d_input){
        for(s_t j=0; j < dl->input_size; ++j){
            float sum = 0.f;
            for(s_t i=0; i < dl->output_size; ++i)
                sum += DL_W_AT(dl, i, j) * (probabilities[i] - (i == target ? 1.f:0.f));
            d_input[j] = sum;
        }
    }

    for(s_t i = 0; i < dl->output_size; ++i){
        float dz = probabilities[i] - (i == target ? 1.f : 0.f);
        dl->d_bias[i] = dz;
        for(s_t j = 0; j < dl->input_size; ++j)
            dl->d_weights[i * dl->input_size + j] = dz * input[j];
    }
    return true;
}

CNNDEF bool CNN_Dense_Update(CNN_DenseLayer *dl, float learning_rate){
    if(!dl || learning_rate <= 0.f){
        _cnn_raise_error(CNN_ERR_INVALID_ARG);
        return false;
    }
    if(!_cnn_valid_dl(dl) || !dl->d_weights || !dl->d_bias){
        _cnn_raise_error(CNN_ERR_INVALID_STATE);
        return false;
    }
    
    for(s_t i = 0; i < dl->output_size * dl->input_size; ++i)
        dl->weights[i] -= learning_rate * dl->d_weights[i];
    for(s_t i = 0; i < dl->output_size; ++i)
        dl->bias[i]    -= learning_rate * dl->d_bias[i];
    return true;
}

CNNDEF bool CNN_TRAIN(
    CNN_ConvLayer  *conv,
    CNN_DenseLayer *dense,
    const CNN_Tensor **images,
    const s_t *labels,
    s_t sample_count,
    s_t epochs,
    float conv_learning_rate,
    float dense_learning_rate,
    float *epoch_losses
){
    if(!conv || !dense || !images || !labels || !sample_count || !epochs ||
        conv_learning_rate <= 0.f || dense_learning_rate <= 0.f){
        _cnn_raise_error(CNN_ERR_INVALID_ARG);
        return false;
    }
    if(!_cnn_valid_cl(conv) || !_cnn_valid_dl(dense) ||
        !conv->d_weights || !conv->d_bias ||
        !dense->d_bias || !dense->d_weights){
        _cnn_raise_error(CNN_ERR_INVALID_STATE);
        return false;
    }

    s_t out_h = 0, out_w = 0;

    if(!images[0] || !_cnn_valid_tensor(images[0]) ||
        images[0]->channels != conv->input_channels){
        _cnn_raise_error(CNN_ERR_SHAPE_MISMATCH);
        return false;
    }

    if(!CNN_Conv_Output_Shape(images[0], conv, &out_w, &out_h)){
        return false;
    }

    s_t feature_count = conv->output_channels * out_h * out_w;

    if(dense->input_size != feature_count){
        _cnn_raise_error(CNN_ERR_SHAPE_MISMATCH);
        return false;
    }

    for(s_t n=0; n<sample_count; ++n){
        if(!images[n] || !_cnn_valid_tensor(images[n])){
            _cnn_raise_error(CNN_ERR_INVALID_STATE);
            return false;
        }
        if(images[n]->width    != images[0]->width    ||
           images[n]->height   != images[0]->height   ||
           images[n]->channels != images[0]->channels ||
           labels[n] >= dense->output_size){
            _cnn_raise_error(CNN_ERR_SHAPE_MISMATCH);
            return false;
        }
    }

    // ALLOCATE TRAINING BUFFERS
    float *flat   = malloc(feature_count      * sizeof *flat);
    float *prob   = malloc(dense->output_size * sizeof *prob);
    float *d_flat = malloc(feature_count      * sizeof *d_flat);
    s_t   *order  = malloc(sample_count       * sizeof *order);

    if(!flat || !prob || !d_flat || !order){
        _cnn_raise_error(CNN_ERR_ALLOC_FAILED);
        goto err;
    }

    for(s_t n=0; n<sample_count; ++n)
        order[n] = n;

    for(s_t epoch = 0; epoch < epochs; ++epoch){
        // SHUFFLE samples
        for(s_t i=sample_count; i > 1; --i){
            s_t j = (s_t)rand() % i;
            s_t tmp = order[i-1];
            order[i-1] = order[j];
            order[j] = tmp;
        }

        float total_loss = 0.f;

        for(s_t n=0; n<sample_count; ++n){
            s_t idx = order[n];
            const CNN_Tensor *img = images[idx];
            s_t target = labels[idx];
            
            // FORWARD: Conv -> ReLU
            if(!CNN_Conv_Forward(img, conv) || !CNN_ReLU(conv->output))
                goto err;

            // Flatten the feature map
            s_t flat_len = 0;
            if(!CNN_Flatten(img, &flat_len, flat))
                goto err;
            if(flat_len != feature_count){
                _cnn_raise_error(CNN_ERR_SHAPE_MISMATCH);
                goto err;
            }

            // FORWARD: Denns -> Softmax
            if(!CNN_Dense_Forward(flat, feature_count, dense) || 
               !CNN_Softmax(dense->output, dense->output_size, prob))
                goto err;
            
            float loss;
            if(!CNN_CrossEntropy(prob, dense->output_size, target, &loss))
                goto err;
            
            total_loss += loss;

            // BACKPROPOGATE BOTH LAYERS

            if(!CNN_Dense_Backward(
                dense, flat, prob, target, d_flat) ||
               !CNN_Conv_Backward(
                img, conv, d_flat, feature_count, NULL))
                goto err;
            
               
            
            // SGD PARAM UPDATE
            if(!CNN_Dense_Update(dense, dense_learning_rate) ||
               !CNN_Conv_Update(conv, conv_learning_rate))
               goto err;
        }
        if(epoch_losses) 
            epoch_losses[epoch] = total_loss / (float)sample_count;
    }
    free(flat);
    free(prob);
    free(d_flat);
    free(order);
    return true;
    err:
        free(flat);
        free(prob);
        free(d_flat);
        free(order);
        return false;
}

#endif // CNN_IMPLEMENTATION
#endif // CNN_H