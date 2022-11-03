import numpy as np

from ml.datasets.hospital import HospitalHoloClean
from ml.plot.old.user_effort_all_potential import PlotterLatex

data = HospitalHoloClean()


fscore_metadata_no_svd_absolute_potential = []
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.090909090909090912, 0.13945578231292519, 0.15015015015015015, 0.2374821173104435, 0.16159860990443092, 0.19914893617021276, 0.25700164744645798, 0.30312750601443467, 0.19972887483054677, 0.22142857142857142, 0.24261138067931184, 0.2670726402783819, 0.2670726402783819, 0.26933101650738489, 0.26933101650738489, 0.25891783567134269, 0.25891783567134269, 0.26422764227642276, 0.26422764227642276, 0.27066450567260941, 0.27066450567260941, 0.27076677316293929, 0.28548895899053633, 0.45193508114856429, 0.51851851851851849, 0.77014925373134324, 0.77014925373134324, 0.78106508875739644, 0.78106508875739644, 0.85862785862785862, 0.85862785862785862, 0.86570247933884303, 0.86687306501547989, 0.89434364994663818, 0.89574468085106396, 0.9015873015873016, 0.9276859504132231, 0.96303696303696307, 0.96303696303696307, 0.96303696303696307, 0.97227722772277225, 0.97227722772277225, 0.98709036742800393, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.090909090909090912, 0.16470588235294117, 0.1753246753246753, 0.26810477657935283, 0.23980222496909762, 0.29868578255675032, 0.37442922374429227, 0.43516483516483517, 0.24377318494965552, 0.26582940868655158, 0.27946295375435104, 0.30641213901125797, 0.30641213901125797, 0.31137140068326014, 0.31137140068326014, 0.31543299467827768, 0.31543299467827768, 0.32283464566929132, 0.32365961633054596, 0.32612966601178789, 0.33104799216454456, 0.33674963396778917, 0.37447698744769875, 0.37447698744769875, 0.37447698744769875, 0.38568588469184889, 0.40476190476190471, 0.40476190476190471, 0.41996911991765312, 0.43367346938775508, 0.43932411674347166, 0.45514445007602639, 0.45514445007602639, 0.45514445007602639, 0.45514445007602639, 0.45670886075949363, 0.45670886075949363, 0.47904191616766473, 0.47904191616766473, 0.47904191616766473, 0.48131539611360247, 0.48584202682563332, 0.5007378258730939, 0.5007378258730939, 0.51859398879266427, 0.51859398879266427, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.090909090909090912, 0.15540540540540543, 0.16190476190476188, 0.25339366515837108, 0.23884514435695542, 0.30555555555555552, 0.38507821901323708, 0.44855491329479774, 0.24726477024070026, 0.26997840172786175, 0.29483767961681745, 0.22630198576245783, 0.22630198576245783, 0.23217618514371036, 0.23217618514371036, 0.23538119911176908, 0.23538119911176908, 0.23997000374953129, 0.23997000374953129, 0.24194756554307115, 0.24739195230998512, 0.25185185185185188, 0.25185185185185188, 0.26976069615663523, 0.27226647356987688, 0.28880866425992779, 0.28880866425992779, 0.30039525691699603, 0.30039525691699603, 0.30039525691699603, 0.31683873264506945, 0.31683873264506945, 0.31683873264506945, 0.2983316977428852, 0.31345826235093699, 0.31575365770670299, 0.32682425488180877, 0.33424283765347884, 0.33424283765347884, 0.45722171113155474, 0.79967819790828631, 0.79967819790828631, 0.79967819790828631, 0.79967819790828631, 0.79967819790828631, 0.79967819790828631, 0.79967819790828631, 0.79967819790828631, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.090909090909090912, 0.16470588235294117, 0.16510903426791276, 0.25481481481481483, 0.22196531791907512, 0.20980392156862746, 0.27573182247403211, 0.32014719411223552, 0.19941916747337851, 0.22179732313575526, 0.24445493157149598, 0.27044609665427505, 0.27044609665427505, 0.27365491651205937, 0.27365491651205937, 0.27885921231326394, 0.27885921231326394, 0.28518859245630174, 0.28518859245630174, 0.32113225963884823, 0.32113225963884823, 0.34296724470134876, 0.34296724470134876, 0.38912133891213391, 0.40991223541559113, 0.42132239876986161, 0.42616249361267239, 0.42937276899541055, 0.43654822335025378, 0.43654822335025378, 0.43654822335025378, 0.45995893223819301, 0.46019517205957883, 0.47219307450157394, 0.47299423177766126, 0.49275362318840582, 0.49275362318840582, 0.49275362318840582, 0.49275362318840582, 0.49275362318840582, 0.49275362318840582, 0.49275362318840582, 0.51521298174442187, 0.51596553471870255, 0.78853601859024014, 0.9922027290448342, 0.9922027290448342, 0.9922027290448342, 0.9922027290448342, 0.9922027290448342, 0.9922027290448342, 0.9922027290448342, 0.9922027290448342, 0.9922027290448342, 0.9922027290448342, 0.9922027290448342, 0.9922027290448342, 0.9922027290448342, 0.9922027290448342, 0.9922027290448342, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
fscore_metadata_no_svd_absolute_potential.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.090909090909090912, 0.12006319115323853, 0.14141414141414144, 0.22589531680440775, 0.23355704697986576, 0.28422425032594523, 0.36724565756823818, 0.41826923076923073, 0.2310712282669658, 0.25649530127142067, 0.28213507625272327, 0.21629855293221631, 0.21629855293221631, 0.21833396728794222, 0.21833396728794222, 0.22371866816311259, 0.22371866816311259, 0.22820318423047767, 0.23356009070294786, 0.23356009070294786, 0.25531914893617019, 0.2585291887793783, 0.27428571428571424, 0.28748068006182381, 0.29275808936825887, 0.29275808936825887, 0.2934154793993069, 0.29135270900609972, 0.29135270900609972, 0.29135270900609972, 0.29135270900609972, 0.30510896748838873, 0.3171337353671515, 0.3171337353671515, 0.43864734299516911, 0.43864734299516911, 0.72931726907630523, 0.73120000000000007, 0.73120000000000007, 0.74444444444444446, 0.80445969125214412, 0.81260647359454852, 0.84335309060118546, 0.84335309060118546, 0.84335309060118546, 0.85016835016835013, 0.85016835016835013, 0.85016835016835013, 0.85016835016835013, 0.85016835016835013, 0.85016835016835013, 0.99123661148977604, 0.99123661148977604, 0.99123661148977604, 0.99123661148977604, 0.99123661148977604, 0.99123661148977604, 0.99123661148977604, 0.99123661148977604, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


nadeef_fscore = 0.05564746578432847
openrefine_fscore = 1.0


dboost_models = ["Gaussian", "Histogram", "Mixture"]
dboost_sizes = [200, 400, 600, 800]
dboost_fscore_all = [
                        # Gaussian
                        [
                            [0.576354679803, 0.382590005373, 0.382590005373, 0.382590005373, 0.576354679803],
                            [0.576354679803, 0.688362919132, 0.729957805907, 0.748917748918, 0.729957805907],
                            [0.729957805907, 0.748917748918, 0.748917748918, 0.748917748918, 0.748917748918],
                            [0.748917748918, 0.748917748918, 0.748917748918, 0.748917748918, 0.729957805907]
                        ],
                        # Histogram
                        [
                            [0.414327202323, 0.3899543379, 0.478968031408, 0.279025624802, 0.478968031408],
                            [0.778188539741, 0.812741312741, 0.478968031408, 0.625557206538, 0.337745687926],
                            [0.812741312741, 0.812741312741, 0.812741312741, 0.812741312741, 0.812741312741],
                            [0.812741312741, 0.812741312741, 0.812741312741, 0.812741312741, 0.812741312741]
                        ],
                        # Mixture
                        [
                            [0.0266940451745, 0.0341637010676, 0.0266940451745, 0.0266940451745, 0.049766718507],
                            [0.0341637010676, 0.0341637010676, 0.0341637010676, 0.0341637010676, 0.0266940451745],
                            [0.0341637010676, 0.0341637010676, 0.0341637010676, 0.0341637010676, 0.0266940451745],
                            [0.0341637010676, 0.0341637010676, 0.0341637010676, 0.0266940451745, 0.0341637010676]
                        ]
                    ]

dboost_matrix_f = np.array(dboost_fscore_all)
dboost_avg_f = np.mean(dboost_matrix_f, axis = 2)

label_potential = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248, 258, 268, 278, 288, 298, 308, 318, 328, 338, 348, 358, 368, 378, 388, 398, 408, 418, 428, 438, 448, 458, 468, 478, 488, 498, 508, 518, 528, 538, 548, 558, 568, 578, 588, 598, 608, 618, 628, 638, 648, 658, 668, 678, 688, 698, 708, 718, 728, 738, 748, 758, 768, 778, 788, 798, 808, 818, 828, 838, 848, 858, 868, 878, 888, 898, 908, 918]



PlotterLatex(data, label_potential, fscore_metadata_no_svd_absolute_potential,
         dboost_models, dboost_sizes, dboost_avg_f,
         nadeef_fscore,
         openrefine_fscore,
         None, xmax=800, filename="Hospital")