This PyTorch SE3 composition layer implementation is being inspired by [Torch gvnn ](https://github.com/ankurhanda/gvnn).  
Thanks for the advice from Ankur Handa @ankurhanda (https://github.com/ankurhanda)(The author of gvnn).

## SE3 composition layer

Purpose: Compose **global pose Tg** with **related pose xi**

 * **Tg** is a SE3 pose represented with 7 parameters (x, y, z, ww, wx, wy, wz)
 * **xi** is a se3 pose represented with 6 parameters (rho1, rho2, rho3, omega_x, omega_y, omega_z)
 * Tutorial of lie group SE3: http://ethaneade.com/lie.pdf


**Txi**: 4x4 Matrix of the exponential mapping of **xi** (lie.pdf Eqation.84)
**Tg_matrix**: Tg represented in matrix form
**T_composed**: The pose calculated by matrix multiplication of Txi and Tg_matrix
```
T_composed = Txi (dot) Tg_matrix
```


## Some implementation story

In [VINet](https://arxiv.org/abs/1701.08376)[1], the author described a important structure called **SE3 composition layer** (Figure.2 in VINet).  However, they do not describe this structure in detail.  I found some related statement in [2] page.9(which is a journal paper from the same advisor).  In this journal paper, I found that gvnn[3] might be their reference of implementing SE3 composition layer.

Unfortunately, [the source code of gvnn](https://github.com/ankurhanda/gvnn) is written by torch-lua.  After taking the kind advice from Ankur Handa, I finished implementing this PyTorch SE3 composition layer.


## Reference
 * [1] R.Clark, S.Wang, H.Wen, A.Markham, andN.Trigoni, “VINet: Visual-Inertial Odometry as a Sequence-to-Sequence Learning Problem,” pp. 3995–4001, 2017.
 * [2] S.Wang, R.Clark, H.Wen, andN.Trigoni, “End-to-end, sequence-to-sequence probabilistic visual odometry through deep neural networks,” Int. J. Rob. Res., 2017.
 * [3] A.Handa, M.Bloesch, V.Pătrăucean, S.Stent, J.McCormac, andA.Davison, “Gvnn: Neural network library for geometric computer vision,” Lect. Notes Comput. Sci. (including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics), vol. 9915 LNCS, pp. 67–82, 2016.
