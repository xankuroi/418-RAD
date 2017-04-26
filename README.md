# GPU-Accelerated Source Engine Radiosity Compilation
**Ryan Lam (rwl) and Lucy Tan (jltan)**

## CHECKPOINT
Most of the time up to this point was spent reverse-engineering the BSP format that the Source Engine uses. Since that needs to be completed before we progress, we haven't done much else, so we're about a week behind our initially proposed schedule. The main challenge has been to figure out how to decode and encode the file format so that the engine will recognize the file for what it is and render accordingly.

What remains is actually implementing and parallelizing the radiosity algorithm. We should be able to do it within the time left.

Please see the [new schedule](#new-schedule) for an updated schedule.

## SUMMARY
We are going to use GPUs to accelerate the Source Engine’s radiosity compilation.

## BACKGROUND
The Source Engine, developed by Valve Software, is an older game engine most notably used by AAA first-person video games, such as Half-Life 2, Counter-Strike: Global Offensive, Portal, and Titanfall. An interesting feature of the Source Engine is that levels require a separate compilation phase-- from human-readable text format to binary format-- before they can be loaded into the engine. Easily the longest part of this compilation phase is the precompiled static lighting phase, in which the compile tools pre-simulate light bounces in the level geometry and bake this information into static lightmaps in the binary format itself. As a result, Source Engine level designers currently need to spend long hours waiting for the static lighting phase to complete before they can test their levels in-game, greatly hindering productivity and workflow.

Currently, the standard Valve Radiosity compilation tool, VRAD, runs entirely on the CPU (hogging all available execution contexts by default, annoyingly enough), and Valve seemingly has no intention of re-writing it to be GPU-accelerated instead. At one point, they implemented their own version of an MPI protocol (“VMPI”) for distributing the lighting work among multiple workstations, but this feature has been broken for the last ten years’ worth of engine versions. So, we figure we will be helping them out a bit.

## THE CHALLENGE
Much of the challenge will be integrating with a system that we do not have extensive knowledge of. Some of the Source Engine’s BSP format may need to be reverse-engineered. Lighting is also a new area for the both of us.

## RESOURCES 
Since this is an extension to the Source Engine, of course we will be referring to the source code for that. We will be testing our code on the GHC GTX 1080. There will certainly be much documentation read about the Source Engine and also radiosity.

## GOALS AND DELIVERABLES
We’re hoping to speed up compilation time for the static lighting phase. We will be showing graphs to illustrate the differences in compilation times between VRAD and our compilation tool. It’s not particularly interesting to wait for the computer to finish crunching the numbers, so there will not be a live demo.

Ideally, our radiosity compilation tool will be significantly faster than Valve’s but we’re aiming just for a small speed up at the moment. 


## PLATFORM CHOICE
As most computer games, including Source Engine games, happen to be for Windows, we thought this would be the best operating system to be running on. The GTX 1080 is a relatively common GPU for a game maker to have available to them, so that’s why we’re testing on them..

## SCHEDULE
### Original Schedule
- Week 1: Familiarize self with the engine and the format
- Week 2: Begin porting to GPU
- Week 3: Finish moving to GPU
- Week 4: Optimize, make graphs

### Completed Tasks
- As of April 25th: file format mostly backwards-engineered

### New Schedule
- April 29th: Finish with BSP (rwl), Put down a basic radiosity function (jltan)
- May 3rd: Parallelization of radiosity (both)
- May 7th: Further optimizations (both)
- May 11th: Finalize Program, Complete Project Presentation (both)
- May 12th: Project Presentation and Final Write Up (both)
