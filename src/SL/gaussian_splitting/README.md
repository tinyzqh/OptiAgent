


# part3.py

```bash
conda install -c conda-forge gxx_linux-64=9.5.0
# 让 PyTorch 用 conda 自带的编译器：
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

python /home/labliu/hezhiqiang/OptiAgent/src/SL/gaussian_splitting/part3.py
```


# Reference

- [A Python Engineer’s Introduction to 3D Gaussian Splatting (Part 1)](https://medium.com/data-science/a-python-engineers-introduction-to-3d-gaussian-splatting-part-1-e133b0449fc6)
- [A Python Engineer’s Introduction to 3D Gaussian Splatting (Part 2)](https://medium.com/data-science/a-python-engineers-introduction-to-3d-gaussian-splatting-part-2-7e45b270c1df)
- [A Python Engineer’s Introduction to 3D Gaussian Splatting (Part 3)](https://medium.com/data-science/a-python-engineers-introduction-to-3d-gaussian-splatting-part-3-398d36ccdd90)