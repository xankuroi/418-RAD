# GPU-Accelerated Source Engine Lightmap Compilation
**Ryan Lam (rwl) and Lucy Tan (jltan)**

## SUMMARY
We accelerated lightmap compilation using the GPU. Valve's lightmap compiler, VRAD, is resource heavy and basically turns the user's computer into a potato while it processes everything, so it's a boon that 418RAD not only is on the GPU, but it's also faster. Now you can use your computer for other things while waiting for the lightmap to finish [compiling](https://imgs.xkcd.com/comics/compiling.png)!

## BACKGROUND
Radiosity is a global illumination algorithm which simulates the effects of real light. Real-time radiosity computations are expensive, so many game engines pre-compute radiosity and overlay the resulting lightmap across their maps to give the sense of somewhat realistic lighting. As one might expect, the computation of the lightmap is quite costly as well. When buiding in the Source Engine, the lightmap compilation can easily take hours and become a bottleneck.

There are a few different ways to handle radiosity. We decided to use the gathering method as opposed to the shooting method, since with gathering, the only shared update was to the gatherer, as opposed with the shooting method where an update would have to be written to all incident patches. The assumption was that we would be able to parallelize across all patches with the gathering method for good performance, as opposed to the shooting method where we could only do shots from one patch at a time. Parallelizing across the patches seemed to be the obvious answer, and terribly data-parallel. 

This algorithm took in an array of patches and updated the patches with the correct information. Creating and updating this array is highly important throughout the code, as all patch information is stored there and will eventually need to be returned. 

## APPROACH
As this project was intended to supplement the Source Engine, much of the code is based off the Source Engine code. We decided to use CUDA since people developing for Source Engine usually have some kind of graphics card available in their machine, as the typical gamer tends to have discrete graphics cards. 

Most of the classes were modeled after the Source Engine's since we needed to read and store data in a file format specific to the Source Engine.

After importing all the data into the GPU, the data required for radiosity is transfered from BSP to CUDABSP. From there, various manipulations are made depending on the phase. 

For the direct lighting phase, [[]]. 

For the light bouncing phase, each face is first placed into an array for easy access. Then each face is divided into patches. These patches are put into a massive array, and the associated face is given a pointer to its first patch. Gathering, the parallel part of this business, begins afterwards. Each patch calculates the amount of light that is reflected to it from all other patches. It updates its color value accordingly. This repeats until the iteration cap is reached or the values converge. We parallelize over the patches, and sync between iterations. After all light data is collected, one final process takes all the data and updates CUDABSP, which will later be translated into BSP and written to file. 



## RESULTS 
Our original plan was to include light bouncing radiosity computations, but unfortunately the version rolled out does not have this functionality properly implemented. However, the direct lighting implementation provides lightmaps comparable to the Source Engine's VRAD.

![VRAD and 418RAD lightmaptest][pic1]
lightmaptest comparison -- VRAD on left, 418RAD on right


![VRAD and 418RAD lightmaptest_hires][pic2]
lightmaptest_hires comparison -- VRAD on left, 418RAD on right


![VRAD and 418RAD ljm][pic3]
ljm comparison -- VRAD on left, 418RAD on right

There are some differences in the shadows due to the differences in how geometry is interpreted in the respective lightmap generators.

![Direct lighting graph comparison][graph]

The baseline was performed on Ryan's laptop, which has an Intel i7 CPU, 2.4 GHZ, 4 cores (8 logical cores). The GTX 850M GPU benchmark was performed on the same machine; the GTX 1080 tests were conducted on the GHC machines. lightmaptest is a very simple map; just a room with a few objects and lights placed into it. lightmaptest_hires is the same map but with higher light sampling density, which is why the shadows are much sharper. ljm is part of a map from an actual game.

As can be seen, compilation time is almost always better on GPU. The exception is with lightmaptest on the GTX 850M; we assume the lengthier time to be from overhead. Speedup varies depending on the map, as there are a multitude of factors that affect how well the lighting alogirhtm does, so it's not easy to tie the increased speed to any particular attribute, such as file size or number of models in the map. However, in general, the GTX 1080's achieve 2x speedup or better.

In builds that had included the flawed bouncing light, the light bouncing computations took up the majority of the time, but with only direct lighting, 

[[]]

Further improvements could be made. Valve's code has a lot of indirection. Since we essentially ported Valve's classes into CUDA, our code, too, has a lot of indirection, so we are missing out on the benefits of locality.

## REFERENCES
[Source SDK Documentation](https://developer.valvesoftware.com/wiki/SDK_Docs)

[SIGGRAPH Radiosity docs](https://www.siggraph.org/education/materials/HyperGraph/radiosity/radiosity.htm)

## WORKSPLIT 
Ryan: Reverse engineering file format, file read and write, direct and ambient lighting

Lucy: Light bouncing

[pic1]: http://imgur.com/ML0nPPC.png
[pic2]: http://imgur.com/mOd9K3G.png
[pic3]: http://imgur.com/vE8YAQx.png
[graph]: http://imgur.com/xjC9zlp.png
