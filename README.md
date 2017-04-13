# BEGAN: Boundary Equilibrium Generative Adversarial Networks
Implementation of Google Brain's [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/pdf/1610.07629v2.pdf) in Tensorflow. \
BEGAN is the state of the art when it comes to generating realistic faces.

<p>
<img src="Result/gamma_0.3.bmp" width="270" height="270" />
<img src="Result/gamma_0.4.bmp" width="270" height="270" />
<img src="Result/gamma_0.5.bmp" width="270" height="270" />
</p>

Figure1. This is random result from my train model. From gamma 0.3 to 0.7. No cherry picking. but 

# Train Progress
This train model is 64x64. 128x128 will be update. Different with original paper is train loss update method, learning rate decay and Nz, Nh size. First, paper's loss update way is Loss_G and Loss_D simultaneously. But when I tried that way, models are mode collapse. So, This code use altenative way. Second, learning rate decay is 0.95 every 2000 iter. This parameter is just train experienc. You can change or see the paper. Last, Nz, Nh is 128. 64 also show nice result, but 128 make much better result. See Figure1 middle result. Under is my train progess.

<p>
<img src="Result/kt.jpg" width="810" height="270" />
</p>

Figure2. Kt graph. When you train model reference this result. It doesn't reach to 1.0. In my case, it's convergece to 0.16

<p>
<img src="Result/m_global.jpg" width="810" height="270" />
</p>

Figure3. Convergence measure(M_global). Simmilar with paper's graph

<p>
<img src="Result/gamma_0.4.bmp" width="270" height="270" />
<img src="Result/decoder.bmp" width="270" height="270" />
</p>

Figure4. Compare with Generator and Decoder. 
