(Code will be uploaded 4/14)
# BEGAN: Boundary Equilibrium Generative Adversarial Networks
Implementation of Google Brain's [BEGAN: Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/pdf/1610.07629v2.pdf) in Tensorflow. \
BEGAN is the state of the art when it comes to generating realistic faces.

<p>
<img src="Result/gamma_0.3.bmp" width="250" height="250" />
<img src="Result/gamma_0.4.bmp" width="350" height="350" />
<img src="Result/gamma_0.5.bmp" width="250" height="250" />
</p>

Figure1a. This is random result from my train model. From gamma 0.3 to 0.5. No cherry picking. gamma 0.3, nice but bias to women's face. gamma 0.4, Best. gamma 0.5, good texture but hole problem. 

<p>
<img src="Result/result.gif" width="300" height="300" />
</p>

Figure1b. From scratch to 200k iter

## Implementation detail 
This train model is 64x64. 128x128 will be update. Different with original paper is train loss update method, learning rate decay and Nz, Nh size. First, paper's loss update way is Loss_G and Loss_D simultaneously. But when I tried that way, models are mode collapse. So, This code use altenative way. Second, learning rate decay is 0.95 every 2000 iter. This parameter is just train experienc. You can change or see the paper. Last, Nz, Nh is 128. 64 also show nice result, but 128 make much better result. See Figure1 middle result. Under is my train progess.

## Train progress
<p>
<img src="Result/kt.jpg" width="810" height="270" />
</p>

Figure2. Kt graph. When you train model, reference this result. It doesn't reach to 1.0. In my case, it's converge to 0.08

<p>
<img src="Result/m_global.jpg" width="810" height="270" />
</p>

Figure3. Convergence measure(M_global). Similar with paper's graph

<p>
<img src="Result/gamma_0.4.bmp" width="270" height="270" />
<img src="Result/decoder.bmp" width="270" height="270" />
</p>

Figure4. Compare with Generator output and Decoder output.  



## Usage
### Make Train Data
1. Unzip celebA dataset to 'Data/celeba/raw'
2. Run ' python ./Data/celeba/face_detect.py '
    
### Train (refer the main.py began_cmd)
    ex) 64x64 img | Nz,Nh 128 | gamma 0.5
    python main.py -f 1 -p "began" 
                          -trd "celeba" -tro "crop" -trs 64   
                          -z 128 -em 128 -fn 64  -b 16 -lr 1e-4 -gm 0.5 -g "0"
    
    ex) 128x128 img | Nz,Nh 64 | gamma 0.3                         
    python main.py -f 1 -p "began" 
                          -trd "celeba" -tro "crop" -trs 128
                          -z 64 -em 64 -fn 128  -b 16 -lr 1e-4 -gm 0.3 -g "0"
                          
### Test (refer the main.py and began_cmd)   
    ex) 64x64 img | Nz,Nh 128 | gamma 0.5
    python main.py -f 0 -p "_began_gm0.5_zh128" 
                          -trd "celeba" -tro "crop" -trs 64   
                          -z 128 -em 128 -fn 64  -b 16 -lr 1e-4 -gm 0.5 -g "0"
                          
                          
## Requirements
- TensorFlow 1.0.0
- Python 2.7.12, Opencv, scipy 0.18.1, numpy 1.11.2                          

## Author
Heumi / ckhfight@gmail.com
    
