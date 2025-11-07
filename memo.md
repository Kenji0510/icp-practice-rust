```bash
    Finished `release` profile [optimized] target(s) in 0.04s
     Running `target/release/icp-practice`
Loaded 8648 points from data/input/clipped_Laser_map_5_voxel-01.pcd
Loaded 8648 points from data/input/clipped_rotated_Laser_map_5_voxel-01.pcd
Iteration 1: mean error = 1.0439916464907397
Iteration 2: mean error = 0.601891628480264
Iteration 3: mean error = 0.38416030034441384
Iteration 4: mean error = 0.27603382225850187
Iteration 5: mean error = 0.2157811017377313
Iteration 6: mean error = 0.1726660844672342
Iteration 7: mean error = 0.14088997752415158
Iteration 8: mean error = 0.11377498474754932
Iteration 9: mean error = 0.09457067056132626
Iteration 10: mean error = 0.07812214785789923
Iteration 11: mean error = 0.06411156241602597
Iteration 12: mean error = 0.05316822317234954
Iteration 13: mean error = 0.03608341319236813
Iteration 14: mean error = 0.0044586581065028654
Iteration 15: mean error = 0.0000013663679639036932
Converged at iteration 15
ICP completed in 38.49s
Final aligned source points:
[[-1.3830013306137687, -2.763094705085671, 2.889031986068966],
 [2.7929324260082566, -0.15770319768268062, 3.1194660523612354],
 [-0.8313907430372666, -2.8195444745720812, 3.252139908482416],
 [2.823343927250143, 0.6265968370780625, 3.127711173212221],
 [-1.3558054797988892, -2.1895806220796383, 2.763139006885925],
 ...,
 [0.2078263023261609, -3.6597065857382995, 2.2933606679577094],
 [0.7728007425239349, -3.689777592698387, 2.2245149450333535],
 [1.3892270286627473, -6.135237592828931, 2.545611509448216],
 [2.488079354942265, 3.6424423237915056, 3.0690105201133195],
 [2.1358247998700244, -2.960904059817216, 2.123690228284867]], shape=[8648, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2
Saved aligned points to data/output/icp_aligned_result.pcd
```
![image](./log/icp-result.jpeg)

```bash
sudo apt update
sudo apt install -y build-essential gfortran pkg-config \
  libopenblas-dev liblapacke-dev \
  libfreetype6-dev libfontconfig1-dev

```

```bash
    Finished `release` profile [optimized] target(s) in 13.34s
     Running `target/release/icp-practice`
Loaded 8648 points from data/input/clipped_Laser_map_5_voxel-01.pcd
Loaded 8648 points from data/input/clipped_rotated_Laser_map_5_voxel-01.pcd
Iteration 1: mean error (from 1000 samples) = 1.0464834623582155
Iteration 2: mean error (from 1000 samples) = 0.609215262737433
Iteration 3: mean error (from 1000 samples) = 0.3816289746406624
Iteration 4: mean error (from 1000 samples) = 0.27243088716564345
Iteration 5: mean error (from 1000 samples) = 0.2152007702925228
Iteration 6: mean error (from 1000 samples) = 0.16099589469484907
Iteration 7: mean error (from 1000 samples) = 0.13743164150654025
Iteration 8: mean error (from 1000 samples) = 0.10981857895711765
Iteration 9: mean error (from 1000 samples) = 0.09191812939351476
Iteration 10: mean error (from 1000 samples) = 0.07818369801694519
Iteration 11: mean error (from 1000 samples) = 0.06053797777929556
Iteration 12: mean error (from 1000 samples) = 0.05150612374520945
Iteration 13: mean error (from 1000 samples) = 0.029858870901635654
Iteration 14: mean error (from 1000 samples) = 0.002605250767303638
Iteration 15: mean error (from 1000 samples) = 0.000001373912088804934
Converged at iteration 15
ICP completed in 4.95s
Final aligned source points:
[[-1.3830013419269138, -2.7630947251344193, 2.8890319682524694],
 [2.7929324213327864, -0.15770322971726616, 3.119466049773871],
 [-0.8313907561885355, -2.8195444956943514, 3.252139893291388],
 [2.8233439248541448, 0.6265968049662518, 3.1277111695631215],
 [-1.355805488837968, -2.189580642401637, 2.763138988315861],
 ...,
 [0.20782629110551387, -3.6597066114002685, 2.2933606588373374],
 [0.7728007315313211, -3.689777620135212, 2.224514938559541],
 [1.3892270089664116, -6.135237621595166, 2.5456115095586807],
 [2.4880793617275776, 3.642442292580384, 3.069010510300134],
 [2.1358247914951427, -2.960904091435943, 2.1236902269680455]], shape=[8648, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2
Saved aligned points to data/output/icp_aligned_result.pcd
```
![image](./log/icp-result-random-sampling.jpeg)