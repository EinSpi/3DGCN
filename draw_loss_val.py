import matplotlib.pyplot as plt
import re


text = """Total Parameters model 1: 22352
fit begin
loss at global_step 3452: 2.96913
loss at global_step 3454: 1.52474
loss at global_step 3456: 1.71878
loss at global_step 3458: 1.43902
loss at global_step 3460: 1.17248
Step 3460 validation: pearson_r2_score=0.0506872
loss at global_step 3462: 0.901395
loss at global_step 3464: 1.04506
loss at global_step 3466: 1.08059
loss at global_step 3468: 1.03235
loss at global_step 3470: 0.977785
Step 3470 validation: pearson_r2_score=0.03432
loss at global_step 3472: 0.842227
loss at global_step 3474: 1.01238
loss at global_step 3476: 1.01182
loss at global_step 3478: 0.920512
loss at global_step 3480: 0.974275
Step 3480 validation: pearson_r2_score=0.0613115
loss at global_step 3482: 1.04144
loss at global_step 3484: 0.820451
loss at global_step 3486: 0.990747
loss at global_step 3488: 0.987401
loss at global_step 3490: 1.01511
Step 3490 validation: pearson_r2_score=0.0842085
loss at global_step 3492: 0.948884
loss at global_step 3494: 0.796512
loss at global_step 3496: 1.02251
loss at global_step 3498: 0.979639
loss at global_step 3500: 0.94299
Step 3500 validation: pearson_r2_score=0.102394
loss at global_step 3502: 0.956984
loss at global_step 3504: 0.894769
loss at global_step 3506: 0.755481
loss at global_step 3508: 0.915762
loss at global_step 3510: 0.916313
Step 3510 validation: pearson_r2_score=0.117972
loss at global_step 3512: 0.945495
loss at global_step 3514: 0.936867
loss at global_step 3516: 0.742064
loss at global_step 3518: 0.879548
loss at global_step 3520: 0.901795
Step 3520 validation: pearson_r2_score=0.135074
loss at global_step 3522: 0.938178
loss at global_step 3524: 0.870752
loss at global_step 3526: 0.9447
loss at global_step 3528: 0.728741
loss at global_step 3530: 0.902834
Step 3530 validation: pearson_r2_score=0.152475
loss at global_step 3532: 0.89847
loss at global_step 3534: 0.924587
loss at global_step 3536: 1.01466
loss at global_step 3538: 0.737668
loss at global_step 3540: 0.940848
Step 3540 validation: pearson_r2_score=0.181197
loss at global_step 3542: 0.87225
loss at global_step 3544: 0.882298
loss at global_step 3546: 0.883774
loss at global_step 3548: 0.924935
loss at global_step 3550: 0.681681
Step 3550 validation: pearson_r2_score=0.188645
loss at global_step 3552: 0.847307
loss at global_step 3554: 0.885678
loss at global_step 3556: 0.836714
loss at global_step 3558: 0.868311
loss at global_step 3560: 0.759075
Step 3560 validation: pearson_r2_score=0.198017
loss at global_step 3562: 0.863591
loss at global_step 3564: 0.902323
loss at global_step 3566: 0.79523
loss at global_step 3568: 0.858419
loss at global_step 3570: 0.788283
Step 3570 validation: pearson_r2_score=0.209949
loss at global_step 3572: 0.713368
loss at global_step 3574: 0.849953
loss at global_step 3576: 0.796057
loss at global_step 3578: 0.815691
loss at global_step 3580: 0.847397
Step 3580 validation: pearson_r2_score=0.206502
loss at global_step 3582: 0.69192
loss at global_step 3584: 0.801101
loss at global_step 3586: 0.841029
loss at global_step 3588: 0.832847
loss at global_step 3590: 0.823566
Step 3590 validation: pearson_r2_score=0.220011
loss at global_step 3592: 0.848997
loss at global_step 3594: 0.815471
loss at global_step 3596: 0.826394
loss at global_step 3598: 0.823101
loss at global_step 3600: 0.867252
Step 3600 validation: pearson_r2_score=0.222359
loss at global_step 3602: 0.895509
loss at global_step 3604: 0.690674
loss at global_step 3606: 0.809932
loss at global_step 3608: 0.892744
loss at global_step 3610: 0.853149
Step 3610 validation: pearson_r2_score=0.235382
loss at global_step 3612: 0.855351
loss at global_step 3614: 0.837245
loss at global_step 3616: 0.7436
loss at global_step 3618: 0.855967
loss at global_step 3620: 0.818108
Step 3620 validation: pearson_r2_score=0.228628
loss at global_step 3622: 0.794645
loss at global_step 3624: 0.798945
loss at global_step 3626: 0.703592
loss at global_step 3628: 0.790415
loss at global_step 3630: 0.821986
Step 3630 validation: pearson_r2_score=0.238779
loss at global_step 3632: 0.793181
loss at global_step 3634: 0.860281
loss at global_step 3636: 0.821276
loss at global_step 3638: 0.656518
loss at global_step 3640: 0.76638
Step 3640 validation: pearson_r2_score=0.235361
loss at global_step 3642: 0.753791
loss at global_step 3644: 0.785821
loss at global_step 3646: 0.819136
loss at global_step 3648: 0.711435
loss at global_step 3650: 0.766874
Step 3650 validation: pearson_r2_score=0.231629
loss at global_step 3652: 0.721017
loss at global_step 3654: 0.802519
loss at global_step 3656: 0.817972
loss at global_step 3658: 0.89559
loss at global_step 3660: 0.603076
Step 3660 validation: pearson_r2_score=0.244294
loss at global_step 3662: 0.827768
loss at global_step 3664: 0.773077
loss at global_step 3666: 0.716826
loss at global_step 3668: 0.836299
loss at global_step 3670: 0.674127
Step 3670 validation: pearson_r2_score=0.24493
loss at global_step 3672: 0.73026
loss at global_step 3674: 0.867268
loss at global_step 3676: 0.751636
loss at global_step 3678: 0.721664
loss at global_step 3680: 0.761547
Step 3680 validation: pearson_r2_score=0.247623
loss at global_step 3682: 0.745728
loss at global_step 3684: 0.768881
loss at global_step 3686: 0.741741
loss at global_step 3688: 0.7556
loss at global_step 3690: 0.69858
Step 3690 validation: pearson_r2_score=0.247848
loss at global_step 3692: 0.669325
loss at global_step 3694: 0.805463
loss at global_step 3696: 0.755753
loss at global_step 3698: 0.822589
loss at global_step 3700: 0.743796
Step 3700 validation: pearson_r2_score=0.251332
loss at global_step 3702: 0.723833
loss at global_step 3704: 0.728851
loss at global_step 3706: 0.828518
loss at global_step 3708: 0.695858
loss at global_step 3710: 0.728469
Step 3710 validation: pearson_r2_score=0.244728
loss at global_step 3712: 0.781581
loss at global_step 3714: 0.688229
loss at global_step 3716: 0.705389
loss at global_step 3718: 0.807397
loss at global_step 3720: 0.765631
Step 3720 validation: pearson_r2_score=0.257733
loss at global_step 3722: 0.770852
loss at global_step 3724: 0.729518
loss at global_step 3726: 0.691784
loss at global_step 3728: 0.734306
loss at global_step 3730: 0.793581
Step 3730 validation: pearson_r2_score=0.237285
loss at global_step 3732: 0.772817
loss at global_step 3734: 0.795231
loss at global_step 3736: 0.683305
loss at global_step 3738: 0.713519
loss at global_step 3740: 0.724913
Step 3740 validation: pearson_r2_score=0.258879
loss at global_step 3742: 0.721836
loss at global_step 3744: 0.781156
loss at global_step 3746: 0.784661
loss at global_step 3748: 0.690722
loss at global_step 3750: 0.765866
Step 3750 validation: pearson_r2_score=0.255687
loss at global_step 3752: 0.717063
loss at global_step 3754: 0.73377
loss at global_step 3756: 0.769536
loss at global_step 3758: 0.658329
loss at global_step 3760: 0.786809
Step 3760 validation: pearson_r2_score=0.256942
loss at global_step 3762: 0.843328
loss at global_step 3764: 0.831895
loss at global_step 3766: 0.796622
loss at global_step 3768: 0.755671
loss at global_step 3770: 0.710646
Step 3770 validation: pearson_r2_score=0.253724
loss at global_step 3772: 0.807828
loss at global_step 3774: 0.817122
loss at global_step 3776: 0.802072
loss at global_step 3778: 0.799407
loss at global_step 3780: 0.656389
Step 3780 validation: pearson_r2_score=0.248629
loss at global_step 3782: 0.739306
loss at global_step 3784: 0.789829
loss at global_step 3786: 0.720289
loss at global_step 3788: 0.759604
loss at global_step 3790: 0.83307
Step 3790 validation: pearson_r2_score=0.255137
loss at global_step 3792: 0.645014
loss at global_step 3794: 0.761948
loss at global_step 3796: 0.779571
loss at global_step 3798: 0.750335
loss at global_step 3800: 0.745857
Step 3800 validation: pearson_r2_score=0.260364
loss at global_step 3802: 0.638297
loss at global_step 3804: 0.718225
loss at global_step 3806: 0.690437
loss at global_step 3808: 0.774006
loss at global_step 3810: 0.764737
Step 3810 validation: pearson_r2_score=0.258138
loss at global_step 3812: 0.772304
loss at global_step 3814: 0.641058
loss at global_step 3816: 0.774027
loss at global_step 3818: 0.739488
loss at global_step 3820: 0.699129
Step 3820 validation: pearson_r2_score=0.261688
loss at global_step 3822: 0.738889
loss at global_step 3824: 0.647553
loss at global_step 3826: 0.730458
loss at global_step 3828: 0.778558
loss at global_step 3830: 0.767972
Step 3830 validation: pearson_r2_score=0.248859
loss at global_step 3832: 0.744671
loss at global_step 3834: 0.763337
loss at global_step 3836: 0.691868
loss at global_step 3838: 0.82715
loss at global_step 3840: 0.756661
Step 3840 validation: pearson_r2_score=0.253918
loss at global_step 3842: 0.736783
loss at global_step 3844: 0.813571
loss at global_step 3846: 0.669265
loss at global_step 3848: 0.7607
loss at global_step 3850: 0.782965
Step 3850 validation: pearson_r2_score=0.253954
loss at global_step 3852: 0.69385
loss at global_step 3854: 0.782884
loss at global_step 3856: 0.779035
loss at global_step 3858: 0.576111
loss at global_step 3860: 0.759704
Step 3860 validation: pearson_r2_score=0.249837
loss at global_step 3862: 0.766138
loss at global_step 3864: 0.776988
loss at global_step 3866: 0.694343
loss at global_step 3868: 0.605033
loss at global_step 3870: 0.725046
Step 3870 validation: pearson_r2_score=0.258905
loss at global_step 3872: 0.788334
loss at global_step 3874: 0.711008
loss at global_step 3876: 0.770552
loss at global_step 3878: 0.764844
loss at global_step 3880: 0.60995
Step 3880 validation: pearson_r2_score=0.260705
loss at global_step 3882: 0.760556
loss at global_step 3884: 0.791555
loss at global_step 3886: 0.796712
loss at global_step 3888: 0.809888
loss at global_step 3890: 0.657747
Step 3890 validation: pearson_r2_score=0.254127
loss at global_step 3892: 0.839993
loss at global_step 3894: 0.748206
loss at global_step 3896: 0.830421
loss at global_step 3898: 0.749133
loss at global_step 3900: 0.775286
Step 3900 validation: pearson_r2_score=0.246814
loss at global_step 3902: 0.64495
loss at global_step 3904: 0.7409
loss at global_step 3906: 0.744311
loss at global_step 3908: 0.818826
loss at global_step 3910: 0.713597
Step 3910 validation: pearson_r2_score=0.260793
loss at global_step 3912: 0.695924
loss at global_step 3914: 0.728234
loss at global_step 3916: 0.728391
loss at global_step 3918: 0.721075
loss at global_step 3920: 0.769749
Step 3920 validation: pearson_r2_score=0.250307
loss at global_step 3922: 0.740797
loss at global_step 3924: 0.590603
loss at global_step 3926: 0.743556
loss at global_step 3928: 0.740729
loss at global_step 3930: 0.731753
Step 3930 validation: pearson_r2_score=0.261103
loss at global_step 3932: 0.753144
loss at global_step 3934: 0.636564
loss at global_step 3936: 0.751815
loss at global_step 3938: 0.762961
loss at global_step 3940: 0.749794
Step 3940 validation: pearson_r2_score=0.257139
loss at global_step 3942: 0.765822
loss at global_step 3944: 0.674397
loss at global_step 3946: 0.610612
loss at global_step 3948: 0.776574
loss at global_step 3950: 0.765591
Step 3950 validation: pearson_r2_score=0.256695
loss at global_step 3952: 0.69534
loss at global_step 3954: 0.732345
loss at global_step 3956: 0.63322
loss at global_step 3958: 0.724764
loss at global_step 3960: 0.808429
Step 3960 validation: pearson_r2_score=0.261083
loss at global_step 3962: 0.752635
loss at global_step 3964: 0.735048
loss at global_step 3966: 0.781483
loss at global_step 3968: 0.635512
loss at global_step 3970: 0.784294
Step 3970 validation: pearson_r2_score=0.262032
loss at global_step 3972: 0.744376
loss at global_step 3974: 0.788093
loss at global_step 3976: 0.769348
loss at global_step 3978: 0.696486
loss at global_step 3980: 0.749759
Step 3980 validation: pearson_r2_score=0.255635
loss at global_step 3982: 0.80672
loss at global_step 3984: 0.824266
loss at global_step 3986: 0.699673
loss at global_step 3988: 0.713738
loss at global_step 3990: 0.66625
Step 3990 validation: pearson_r2_score=0.247338
loss at global_step 3992: 0.725235
loss at global_step 3994: 0.739597
loss at global_step 3996: 0.730623
loss at global_step 3998: 0.732632
loss at global_step 4000: 0.636322
Step 4000 validation: pearson_r2_score=0.256069
loss at global_step 4002: 0.740465
loss at global_step 4004: 0.741386
loss at global_step 4006: 0.779552
loss at global_step 4008: 0.735438
loss at global_step 4010: 0.731544
Step 4010 validation: pearson_r2_score=0.25412
loss at global_step 4012: 0.64578
loss at global_step 4014: 0.745447
loss at global_step 4016: 0.782125
loss at global_step 4018: 0.715845
loss at global_step 4020: 0.735539
Step 4020 validation: pearson_r2_score=0.261393
loss at global_step 4022: 0.615592
loss at global_step 4024: 0.738397
loss at global_step 4026: 0.737161
loss at global_step 4028: 0.727393
loss at global_step 4030: 0.755463
Step 4030 validation: pearson_r2_score=0.252636
loss at global_step 4032: 0.746808
loss at global_step 4034: 0.658425
loss at global_step 4036: 0.713835
loss at global_step 4038: 0.739454
loss at global_step 4040: 0.760606
Step 4040 validation: pearson_r2_score=0.259606
loss at global_step 4042: 0.740794
loss at global_step 4044: 0.591601
loss at global_step 4046: 0.729767
loss at global_step 4048: 0.717083
loss at global_step 4050: 0.807836
Step 4050 validation: pearson_r2_score=0.257511
loss at global_step 4052: 0.726243
loss at global_step 4054: 0.69517
loss at global_step 4056: 0.621654
loss at global_step 4058: 0.810284
loss at global_step 4060: 0.666004
Step 4060 validation: pearson_r2_score=0.263928
loss at global_step 4062: 0.783504
loss at global_step 4064: 0.764265
loss at global_step 4066: 0.589149
loss at global_step 4068: 0.719028
loss at global_step 4070: 0.73612
Step 4070 validation: pearson_r2_score=0.251338
loss at global_step 4072: 0.720164
loss at global_step 4074: 0.733803
loss at global_step 4076: 0.732447
loss at global_step 4078: 0.626418
loss at global_step 4080: 0.766014
Step 4080 validation: pearson_r2_score=0.256472
loss at global_step 4082: 0.705245
loss at global_step 4084: 0.665974
loss at global_step 4086: 0.741132
loss at global_step 4088: 0.633269
loss at global_step 4090: 0.678982
Step 4090 validation: pearson_r2_score=0.254868
loss at global_step 4092: 0.749025
loss at global_step 4094: 0.73442
loss at global_step 4096: 0.743031
loss at global_step 4098: 0.732288
loss at global_step 4100: 0.59948
Step 4100 validation: pearson_r2_score=0.260373
loss at global_step 4102: 0.724121
loss at global_step 4104: 0.687455
loss at global_step 4106: 0.74628
loss at global_step 4108: 0.740037
loss at global_step 4110: 0.672891
Step 4110 validation: pearson_r2_score=0.264059
loss at global_step 4112: 0.732325
loss at global_step 4114: 0.73906
loss at global_step 4116: 0.773419
loss at global_step 4118: 0.725984
loss at global_step 4120: 0.752713
Step 4120 validation: pearson_r2_score=0.260314
loss at global_step 4122: 0.661452
loss at global_step 4124: 0.730359
loss at global_step 4126: 0.776145
loss at global_step 4128: 0.737435
loss at global_step 4130: 0.695031
Step 4130 validation: pearson_r2_score=0.259502
loss at global_step 4132: 0.56827
loss at global_step 4134: 0.728245
loss at global_step 4136: 0.736643
loss at global_step 4138: 0.764151
loss at global_step 4140: 0.738595
Step 4140 validation: pearson_r2_score=0.257265
loss at global_step 4142: 0.662047
loss at global_step 4144: 0.602177
loss at global_step 4146: 0.667281
loss at global_step 4148: 0.752666
loss at global_step 4150: 0.718298
Step 4150 validation: pearson_r2_score=0.261117
loss at global_step 4152: 0.71836
loss at global_step 4154: 0.63772
loss at global_step 4156: 0.692836
loss at global_step 4158: 0.79812
loss at global_step 4160: 0.667117
Step 4160 validation: pearson_r2_score=0.262698
loss at global_step 4162: 0.749862
loss at global_step 4164: 0.686675
loss at global_step 4166: 0.654545
loss at global_step 4168: 0.695418
loss at global_step 4170: 0.718496
Step 4170 validation: pearson_r2_score=0.26273
loss at global_step 4172: 0.690286
loss at global_step 4174: 0.792248
loss at global_step 4176: 0.615147
loss at global_step 4178: 0.740213
loss at global_step 4180: 0.733377
Step 4180 validation: pearson_r2_score=0.256147
loss at global_step 4182: 0.691018
loss at global_step 4184: 0.765156
loss at global_step 4186: 0.694354
loss at global_step 4188: 0.604837
loss at global_step 4190: 0.741515
Step 4190 validation: pearson_r2_score=0.258556
loss at global_step 4192: 0.714645
loss at global_step 4194: 0.748599
loss at global_step 4196: 0.659057
loss at global_step 4198: 0.61365
loss at global_step 4200: 0.704093
Step 4200 validation: pearson_r2_score=0.258211
loss at global_step 4202: 0.747336
loss at global_step 4204: 0.730296
loss at global_step 4206: 0.731248
loss at global_step 4208: 0.74707
loss at global_step 4210: 0.594995
Step 4210 validation: pearson_r2_score=0.260234
loss at global_step 4212: 0.822882
loss at global_step 4214: 0.701283
loss at global_step 4216: 0.668953
loss at global_step 4218: 0.727738
loss at global_step 4220: 0.596737
Step 4220 validation: pearson_r2_score=0.240169
loss at global_step 4222: 0.716092
loss at global_step 4224: 0.75509
loss at global_step 4226: 0.681398
loss at global_step 4228: 0.729911
loss at global_step 4230: 0.69524
Step 4230 validation: pearson_r2_score=0.257604
loss at global_step 4232: 0.623733
loss at global_step 4234: 0.701691
loss at global_step 4236: 0.725893
loss at global_step 4238: 0.762504
loss at global_step 4240: 0.745534
Step 4240 validation: pearson_r2_score=0.262576
loss at global_step 4242: 0.586417
loss at global_step 4244: 0.731093
loss at global_step 4246: 0.712873
loss at global_step 4248: 0.767767
loss at global_step 4250: 0.658253
Step 4250 validation: pearson_r2_score=0.250938
loss at global_step 4252: 0.722764
loss at global_step 4254: 0.567186
loss at global_step 4256: 0.712302
loss at global_step 4258: 0.681431
loss at global_step 4260: 0.795126
Step 4260 validation: pearson_r2_score=0.260954
loss at global_step 4262: 0.695414
loss at global_step 4264: 0.626417
loss at global_step 4266: 0.720884
loss at global_step 4268: 0.690077
loss at global_step 4270: 0.76313
Step 4270 validation: pearson_r2_score=0.263326
loss at global_step 4272: 0.705132
loss at global_step 4274: 0.695087
loss at global_step 4276: 0.59023
loss at global_step 4278: 0.755271
loss at global_step 4280: 0.689355
Step 4280 validation: pearson_r2_score=0.25965
loss at global_step 4282: 0.708337
loss at global_step 4284: 0.724697
loss at global_step 4286: 0.608901
loss at global_step 4288: 0.673429
loss at global_step 4290: 0.72193
Step 4290 validation: pearson_r2_score=0.25407
loss at global_step 4292: 0.716576
loss at global_step 4294: 0.732791
loss at global_step 4296: 0.735856
loss at global_step 4298: 0.596104
loss at global_step 4300: 0.750618
Step 4300 validation: pearson_r2_score=0.2607
loss at global_step 4302: 0.71224
loss at global_step 4304: 0.770765
loss at global_step 4306: 0.677513
loss at global_step 4308: 0.637445
loss at global_step 4310: 0.770273
Step 4310 validation: pearson_r2_score=0.258623
loss at global_step 4312: 0.719587
loss at global_step 4314: 0.714916
loss at global_step 4316: 0.725709
loss at global_step 4318: 0.655257
loss at global_step 4320: 0.610012
Step 4320 validation: pearson_r2_score=0.257682
loss at global_step 4322: 0.802812
loss at global_step 4324: 0.783582
loss at global_step 4326: 0.754084
loss at global_step 4328: 0.725614
loss at global_step 4330: 0.604539
Step 4330 validation: pearson_r2_score=0.247892
loss at global_step 4332: 0.70949
loss at global_step 4334: 0.769903
loss at global_step 4336: 0.687813
loss at global_step 4338: 0.713512
loss at global_step 4340: 0.749762
Step 4340 validation: pearson_r2_score=0.260699
loss at global_step 4342: 0.577847
loss at global_step 4344: 0.723623
loss at global_step 4346: 0.711086
loss at global_step 4348: 0.689493
loss at global_step 4350: 0.724113
Step 4350 validation: pearson_r2_score=0.25971
loss at global_step 4352: 0.613047
loss at global_step 4354: 0.716346
loss at global_step 4356: 0.788752
loss at global_step 4358: 0.709882
loss at global_step 4360: 0.634597
Step 4360 validation: pearson_r2_score=0.257574
loss at global_step 4362: 0.717641
loss at global_step 4364: 0.604484
loss at global_step 4366: 0.756473"""

# 使用正则表达式查找特定模式（'$'后的数字）
# '\$' 表示字面意义上的 '$' 符号, '\d+\.\d+' 匹配数字
matches = re.findall(r'\=(\d+\.\d+)', text)

# 将字符串转换为浮点数
pearson_r2_scores = [float(num) for num in matches]


# Creating a list of step numbers corresponding to each score
steps = range(3460, 4370, 10)

# Creating the plot
plt.figure(figsize=(10, 5))
plt.plot(steps, pearson_r2_scores, marker='o')

# Adding title and labels
plt.title('Pearson R2 Scores Over Steps of model with params %d'%22352)
plt.xlabel('Step')
plt.ylabel('Pearson R2 Score')

# Displaying the plot
plt.show()
