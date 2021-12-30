# MoRE

## Introduction


This folder includes the code of running MoRE, MoRE(all), SA, SA(all), and PGD, Deepfool Fog, Snow, and Rotation attack.

## Running the code (cifar)

###For MoRE and MoRE(all)
```
python main_desk_adv.py --training True --testing False --method 'more' --num_experts = 4  --checkpoint_loc './checkpoint/MoRE_adv.pth' ### adv MoRE training

python main_desk_all.py --testing True --testing False --method 'more' --num_experts = 7  --checkpoint_loc './checkpoint/MoRE_all.pth'   ### all MoRE training
```
###For SA and SA(all)
```
python main_desk_adv.py --training True --testing False --method 'base' --num_experts = 4  --checkpoint_loc './checkpoint/SA_adv.pth' ### adv MoRE training

python main_desk_all.py --testing True --testing False --method 'base' --num_experts = 7  --checkpoint_loc './checkpoint/SA_all.pth'   ### all MoRE training
```

###For deepfool (MoRE_adv)
```
python main_desk_adv.py --training False --testing True --method 'more' --usemodel 'more' --num_experts = 4  --checkpoint_loc './checkpoint/MoRE_adv.pth' --epochs 1 --batch_size 1 ### adv MoRE training

```

###For Tiny-ImageNet

add argument --dataset tinyimagenet