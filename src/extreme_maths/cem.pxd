cdef extern from "../../c/include/extreme_maths.h":
    void init()
    void clean()

    ctypedef struct Vector_float:
        int size
        float* array

    void iadd_f(float *a, float *b, int *s)
    void isub_f(float *a, float *b, int *s)
    void imul_f(float *a, float *b, int *s)
    void idiv_f(float *a, float *b, int *s)

    float* add_f(float *a, float *b, int *s)
    float* sub_f(float *a, float *b, int *s)
    float* mul_f(float *a, float *b, int *s)
    float* div_f(float *a, float *b, int *s)

    void iadd_v_f(Vector_float *a, Vector_float *b)
    void isub_v_f(Vector_float *a, Vector_float *b)
    void imul_v_f(Vector_float *a, Vector_float *b)
    void idiv_v_f(Vector_float *a, Vector_float *b)

    Vector_float add_v_f(Vector_float *a, Vector_float *b)
    Vector_float sub_v_f(Vector_float *a, Vector_float *b)
    Vector_float mul_v_f(Vector_float *a, Vector_float *b)
    Vector_float div_v_f(Vector_float *a, Vector_float *b)

    float* result_f(Vector_float *vec)
    float* clean_f(Vector_float *vec)

    Vector_float copy_f(float *arr, int size)
    Vector_float point_f(float *arr, int size)
