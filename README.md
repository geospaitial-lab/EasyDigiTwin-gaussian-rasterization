# Differential Gaussian Rasterization

**NOTE**: This is a modified version of [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) incorporating changes from [Mip-Splatting](https://github.com/autonomousvision/mip-splatting/tree/main), [AbsGS](https://github.com/TY424/AbsGS/tree/main), [RaDe-GS](https://github.com/BaowenZ/RaDe-GS/tree/main) nas [MCMC-GS](https://github.com/ubc-vision/3dgs-mcmc/tree/main).
It also adds additional gradient computations for camera intrinsics.


Used as the rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". If you can make use of it in your own research, please be so kind to cite us.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>
