

# File Structure

```bash
Assignment03/
├── include/
│   └── exp_integral.h
├── src/
│   ├── main.cpp
│   ├── exp_integral_cpu.cpp
│   └── exp_integral_gpu.cu
├── bin/
│   └── exponentialIntegral.out (after build)
├── Makefile
└── README.md
└── Report_Exponential_Itegral_Calculation.pdf
```



# Build Instructions

```bash
make
```

The executable will be placed in `bin/exponentialIntegral.out`

# Run Options

```bash
./bin/exponentialIntegral.out [options]

Options:
  -n N        : compute E_n(x) for n = 1 to N (default: 10)
  -m M        : number of samples for x in [a, b] (default: 10)
  -a value    : start of x interval (default: 0.0)
  -b value    : end of x interval   (default: 10.0)
  -c          : skip CPU execution
  -g          : skip GPU execution
  -t          : print timing results
  -v          : verbose mode (print all values)
  -h          : show help message


```

Example:

```bash
./bin/exponentialIntegral.out -n 500 -m 500 -t
```

![image-20250526024557500](/Users/sunlishuang/Library/Application Support/typora-user-images/image-20250526024557500.png)

