

#include"../include/algebra.h"
#include"../include/tensor.h"

namespace grad {
    int test_arithmetic() {

        int* a_shape = (int*)malloc(sizeof(int)*1);
        a_shape[0] = 4;
        Tensor* a = new Tensor(1, a_shape);
        a->name = "a";
        for(int i = 0; i < a->size(); i++) { a->cpu_ptr()[i] = 1.0f; }

        int* b_shape = (int*)malloc(sizeof(int)*1);
        b_shape[0] = 2;
        Tensor* b = new Tensor(1, b_shape);
        b->name = "b";
        for(int i = 0; i < b->size(); i++) { b->cpu_ptr()[i] = 2.0f; }

        int* c_shape = (int*)malloc(sizeof(int)*1);
        c_shape[0] = 4;
        Tensor* c = new Tensor(1, c_shape);
        c->name = "c";
        for(int i = 0; i < c->size(); i++) { c->cpu_ptr()[i] = 3.0f; }

        int* d_shape = (int*)malloc(sizeof(int)*1);
        d_shape[0] = 4;
        Tensor* d = new Tensor(1, d_shape);
        d->name = "d";
        for(int i = 0; i < d->size(); i++) { d->cpu_ptr()[i] = 4.0f; }

        int* e_shape = (int*)malloc(sizeof(int)*1);
        e_shape[0] = 2;
        Tensor* e = new Tensor(1, e_shape);
        e->name = "e";
        for(int i = 0; i < e->size(); i++) { e->cpu_ptr()[i] = 5.0f; }

        a->toGPU();
        b->toGPU();
        c->toGPU();
        d->toGPU();
        e->toGPU();

        Tensor* ab = a->add(b);
        ab->name = "ab";
        Tensor* cd = c->sub(d);
        cd->name = "cd";
        Tensor* abcd = ab->mul(cd);
        abcd->name = "abcd"; 
        Tensor* f = abcd->div(e);
        f->name = "f";

        f->backward();
        std::cout << f->pretty_grad(true);

        return 0;
    }

    int test_matmul() {
        int* shape_a0 = (int*)malloc(sizeof(int)*2);
        shape_a0[0] = 2;
        shape_a0[1] = 3;
        Tensor* a0 = new Tensor(2, shape_a0);
        a0->name = "a";
        a0->toGPU();
        a0->set(1.0f);

        int* shape_b0 = (int*)malloc(sizeof(int)*2);
        shape_b0[0] = 3;
        shape_b0[1] = 2;
        Tensor* b0 = new Tensor(2, shape_b0);
        b0->name = "b";
        b0->toGPU();
        b0->set(1.0f);

        Tensor* c0 = a0->mm(b0);
        c0->name = "c";
        c0->backward();
        c0-> toCPU();
        std::cout << "Result: " << c0->pretty() << "\n";
        std::cout << "Gradient: " << c0->pretty_grad(true) << "\n";

        std::cout << "\n\n--- a: same col, new row ---\n\n";

        for(int i = 0; i < 2; i++) {
            int* shape_a1 = (int*)malloc(sizeof(int)*2);
            shape_a1[0] = 2;
            shape_a1[1] = 3;
            Tensor* a1 = new Tensor(2, shape_a1);
            a1->name = "a";
            a1->zeros();
            a1->cpu_ptr()[i*3] = 1.0f;
            a1->toGPU();

            int* shape_b1 = (int*)malloc(sizeof(int)*2);
            shape_b1[0] = 3;
            shape_b1[1] = 4;
            Tensor* b1 = new Tensor(2, shape_b1);
            b1->name = "b";
            b1->toGPU();
            b1->set(1.0f);

            Tensor* c1 = a1->mm(b1);
            c1->name = "c";
            c1->backward();
            c1-> toCPU();
            std::cout << "Result: " << c1->pretty() << "\n";
            std::cout << "Gradient: " << c1->pretty_grad(true) << "\n";
        }

        std::cout << "\n\n--- a: same row, new col ---\n\n";

        for(int i = 0; i < 3; i++) {
            int* shape_a1 = (int*)malloc(sizeof(int)*2);
            shape_a1[0] = 2;
            shape_a1[1] = 3;
            Tensor* a1 = new Tensor(2, shape_a1);
            a1->name = "a";
            a1->zeros();
            a1->cpu_ptr()[i] = 1.0f;
            a1->toGPU();

            int* shape_b1 = (int*)malloc(sizeof(int)*2);
            shape_b1[0] = 3;
            shape_b1[1] = 4;
            Tensor* b1 = new Tensor(2, shape_b1);
            b1->name = "b";
            b1->toGPU();
            b1->set(1.0f);

            Tensor* c1 = a1->mm(b1);
            c1->name = "c";
            c1->backward();
            c1-> toCPU();
            std::cout << "Result: " << c1->pretty() << "\n";
            std::cout << "Gradient: " << c1->pretty_grad(true) << "\n";
        }

         std::cout << "\n\n--- b: same col, new row ---\n\n";

        for(int i = 0; i < 3; i++) {
            int* shape_a1 = (int*)malloc(sizeof(int)*2);
            shape_a1[0] = 2;
            shape_a1[1] = 3;
            Tensor* a1 = new Tensor(2, shape_a1);
            a1->name = "a";
            a1->toGPU();
            a1->set(1.0f);

            int* shape_b1 = (int*)malloc(sizeof(int)*2);
            shape_b1[0] = 3;
            shape_b1[1] = 4;
            Tensor* b1 = new Tensor(2, shape_b1);
            b1->name = "b";
            b1->zeros();
            b1->cpu_ptr()[i*4] = 1.0f;
            b1->toGPU();

            Tensor* c1 = a1->mm(b1);
            c1->name = "c";
            c1->backward();
            c1-> toCPU();
            std::cout << "Result: " << c1->pretty() << "\n";
            std::cout << "Gradient: " << c1->pretty_grad(true) << "\n";
        }

        std::cout << "\n\n--- b: same row, new col ---\n\n";

        for(int i = 0; i < 4; i++) {
            int* shape_a1 = (int*)malloc(sizeof(int)*2);
            shape_a1[0] = 2;
            shape_a1[1] = 3;
            Tensor* a1 = new Tensor(2, shape_a1);
            a1->name = "a";
            a1->toGPU();
            a1->set(1.0f);

            int* shape_b1 = (int*)malloc(sizeof(int)*2);
            shape_b1[0] = 3;
            shape_b1[1] = 4;
            Tensor* b1 = new Tensor(2, shape_b1);
            b1->name = "b";
            b1->zeros();
            b1->cpu_ptr()[i] = 1.0f;;
            b1->toGPU();

            Tensor* c1 = a1->mm(b1);
            c1->name = "c";
            c1->backward();
            c1-> toCPU();
            std::cout << "Result: " << c1->pretty() << "\n";
            std::cout << "Gradient: " << c1->pretty_grad(true) << "\n";
        }
        

        return 0;
    }

    int test_index() {
        int* shape_emb = (int*)malloc(sizeof(int)*2);
        shape_emb[0] = 3;
        shape_emb[1] = 3;

        Tensor* emb = new Tensor(2, shape_emb);
        emb->name = "emb";
        for(int i = 0; i < shape_emb[0]; i++) {
            for(int j = 0; j < shape_emb[1]; j++) {
                emb->cpu_ptr()[i*shape_emb[1] + j] = 1.0f*j;
            }
        }
        emb->toGPU();

        int* indecies = (int*)malloc(sizeof(int)*shape_emb[0]);
        for(int i = 0; i < shape_emb[0]; i++) { indecies[i] = i; }
        int* dev_indecies;
        cce(cudaMalloc(&dev_indecies, sizeof(int)*shape_emb[0]));
        cce(cudaMemcpy(dev_indecies, indecies, sizeof(int)*shape_emb[0], cudaMemcpyHostToDevice));

        Tensor* t = emb->index(dev_indecies);
        t->name = "t";

        t->backward();
        
        t->toCPU();
        std::cout << "Result: " << t->pretty() << "\n";
        std::cout << "Gradient: " << t->pretty_grad(true) << "\n";

        return 0;
    }

    int test_loss() {
        int* shape_logits = (int*)malloc(sizeof(int)*2);
        shape_logits[0] = 4;
        shape_logits[1] = 64;
        Tensor* logits = new Tensor(2, shape_logits);
        logits->name = "logits";
        for (int i = 0; i < logits->size(); i++) { logits->cpu_ptr()[i] = 1.0f/logits->size(); }
        logits->toGPU();

        Tensor* L = logits->loss();
        L->backward();

        L->toCPU();
        std::cout << "Result: " << L->pretty() << "\n";
        std::cout << "Gradient: " << L->pretty_grad(true) << "\n";

        return 0;
    }

    int test_softmax() {
        int* shape = (int*)malloc(sizeof(int)*2);
        shape[0] = 3;
        shape[1] = 3;

        Tensor* x = new Tensor(2, shape);
        x->name = "x";
        for(int i = 0; i < shape[0]; i++) {
            for(int j = 0; j < shape[1]; j++) {
                x->cpu_ptr()[i*shape[1] + j] = 1.0f*j;
            }
        }
        x->toGPU();

        Tensor* y = x->softmax(3, 3);
        y->backward();
        
        y->toCPU();
        std::cout << "Result: " << y->pretty() << "\n";
        std::cout << "Gradient: " << y->pretty_grad(true) << "\n";

        return 0;
    }

    int test_split() {
        int* shape = (int*)malloc(sizeof(int)*4);
        shape[0] = 4;
        shape[1] = 3;
        shape[2] = 2;
        shape[3] = 2;
        Tensor* x = new Tensor(4, shape);
        x->name = "x";
        for(int k = 0; k < 4; k++) {
            for(int i = 0; i < 3; i++) {
                for(int j = 0; j < 2; j++) { 
                    x->cpu_ptr()[i*4+j*2 + k*12] = 1.0f*(j+k);
                    x->cpu_ptr()[i*4+j*2+1 + k*12] = 1.0f*(j+k);
                }
            }
        }
        

        std::cout << "x:\n";
        std::cout << x->pretty() << "\n";
        x->toGPU();

        Tensor** ys = x->split(2, 2);

        for(int i = 0; i < 2; i++) {
            ys[i]->toCPU();
            std::cout << "y[" << i << "]\n";
            std::cout << ys[i]->pretty() << "\n"; 
            ys[i]->toGPU();
        }

        int* mul_shape = (int*)malloc(sizeof(int));
        mul_shape[0] = 2;
        Tensor* batched_mul = new Tensor(1, mul_shape);
        batched_mul->name = "multiplicant";
        x->name = "x";
        batched_mul->cpu_ptr()[0] = 1.0f;
        batched_mul->cpu_ptr()[1] = 2.0f;

        batched_mul->toGPU();

        Tensor* y = (ys[1]->mul(batched_mul))->add(ys[0]);
        y->name = "y";

        y->toCPU();
        std::cout << "y:\n";
        std::cout << y->pretty() << "\n";
        y->toGPU();

        batched_mul->toCPU();
        std::cout << "batched_mul:\n";
        std::cout << batched_mul->pretty() << "\n";
        batched_mul->toGPU();

        y->backward();

        y->toCPU();
        std::cout << "y grad:\n";
        std::cout << y->pretty_grad(true) << "\n\n";

        return 0;
    }
}

