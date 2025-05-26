#include "exp_integral.h"
#include <cmath>

float exponentialIntegralFloatCPU(int n, float x) {
    const float epsilon = 1.E-30f;
    const float big = 1.E+30f;
    const float eulerConstant = 0.5772156649015329f;
    int i, ii, nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans;

    if (n == 0) return expf(-x) / x;
    if (x > 1.0f) {
        b = x + n;
        c = big;
        d = 1.0f / b;
        h = d;
        for (i = 1; i <= 1000; i++) {
            a = -i * (nm1 + i);
            b += 2.0f;
            d = 1.0f / (a * d + b);
            c = b + a / c;
            del = c * d;
            h *= del;
            if (fabsf(del - 1.0f) <= epsilon)
                return h * expf(-x);
        }
        return h * expf(-x);
    } else {
        ans = (nm1 != 0) ? 1.0f / nm1 : -logf(x) - eulerConstant;
        fact = 1.0f;
        for (i = 1; i <= 1000; i++) {
            fact *= -x / i;
            if (i != nm1) {
                del = -fact / (i - nm1);
            } else {
                psi = -eulerConstant;
                for (ii = 1; ii <= nm1; ii++) psi += 1.0f / ii;
                del = fact * (-logf(x) + psi);
            }
            ans += del;
            if (fabsf(del) < fabsf(ans) * epsilon) return ans;
        }
        return ans;
  }
}

double exponentialIntegralDoubleCPU(int n, double x) {
    const double epsilon = 1.E-30;
    const double big = 1.E+300;
    const double eulerConstant = 0.5772156649015329;
    int i, ii, nm1 = n - 1;
    double a, b, c, d, del, fact, h, psi, ans;

    if (n == 0) return exp(-x) / x;
    if (x > 1.0) {
        b = x + n;
        c = big;
        d = 1.0 / b;
        h = d;
        for (i = 1; i <= 1000; i++) {
            a = -i * (nm1 + i);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            del = c * d;
            h *= del;
            if (fabs(del - 1.0) <= epsilon)
                return h * exp(-x);
        }
        return h * exp(-x);
    } else {
        ans = (nm1 != 0) ? 1.0 / nm1 : -log(x) - eulerConstant;
        fact = 1.0;
        for (i = 1; i <= 1000; i++) {
            fact *= -x / i;
            if (i != nm1) {
                del = -fact / (i - nm1);
            } else {
                psi = -eulerConstant;
                for (ii = 1; ii <= nm1; ii++) psi += 1.0 / ii;
                del = fact * (-log(x) + psi);
            }
            ans += del;
            if (fabs(del) < fabs(ans) * epsilon) return ans;
        }
        return ans;
    }
}
