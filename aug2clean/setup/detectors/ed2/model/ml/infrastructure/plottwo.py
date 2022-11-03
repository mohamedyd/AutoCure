import matplotlib.pyplot as plt

labels = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248, 258, 268, 278, 288, 298, 308, 318, 328, 338, 348, 358, 368, 378, 388, 398, 408, 418, 428, 438, 448, 458, 468, 478, 488, 498, 508, 518, 528, 538, 548, 558, 568, 578, 588, 598, 608, 618, 628, 638, 648, 658, 668, 678, 688, 698, 708, 718, 728, 738, 748, 758, 768, 778, 788, 798, 808, 818, 828, 838, 848, 858, 868, 878, 888, 898, 908, 918, 928, 938, 948, 958, 968, 978, 988, 998, 1008, 1018, 1028, 1038]


outlier = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.038485272751715295, 0.04300243821310364, 0.05544768528969246, 0.058559603314934115, 0.06942195818319793, 0.06771657785600313, 0.06939526596164271, 0.06944815490618102, 0.06876198469638035, 0.06712222459091947, 0.06711410549498933, 0.06682158036936048, 0.06607124552362979, 0.06653152292648382, 0.06841095093553286, 0.0688947437087868, 0.06929929443046258, 0.07364730388052727, 0.07670765068371027, 0.08122644221224273, 0.08650409795457184, 0.092233076122767, 0.09682289273398623, 0.10220092703491421, 0.10546861047314768, 0.11181425968107608, 0.12045451775363665, 0.12416021048972223, 0.13207171977338822, 0.1384081633098166, 0.14755589325702206, 0.16130682747053987, 0.17861184102098604, 0.18147406946945419, 0.18677651024167546, 0.19521101122807086, 0.20777506578421973, 0.22429607018586267, 0.24306950479183506, 0.2547711276717409, 0.27165513934912966, 0.277860880082437, 0.30698761493045124, 0.3300010646790109, 0.33298877986649844, 0.39180147377141655, 0.40288820328770375, 0.42077751183816936, 0.471038589086121, 0.48288243475698867, 0.4841653528756066, 0.5358651009180072, 0.5431181813751115, 0.5454797511259424, 0.5551917732231619, 0.5551917732231619, 0.5732644738183739, 0.6137169767657268, 0.6137169767657268, 0.6256903802635554, 0.6469235960663731, 0.6469235960663731, 0.6662301592713652, 0.7226780386043389, 0.7227634543281709, 0.7227634543281709, 0.7243479961008931, 0.760319743651588, 0.7821025664325904, 0.8306624451592199, 0.8388570639443534, 0.8877269785601243, 0.8877269785601243, 0.8877554320222231, 0.8877554320222231, 0.9339214817313717, 0.9339214817313717, 0.9339214817313717, 0.9827142382706071, 0.9927408576139968, 0.9927408576139968, 0.9927408576139968, 0.9941352799247536, 0.9941352799247536, 0.9941352799247536, 0.9941352799247536, 0.9941352799247536, 0.9941352799247536, 0.9941352799247536, 0.9941352799247536, 0.9941352799247536, 0.9941352799247536, 0.9941352799247536, 0.9941352799247536, 0.9941352799247536, 0.9941352799247536, 0.9941352799247536]
random = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003885070524431571, 0.003885070524431571, 0.006569812682857938, 0.01439283234615854, 0.04485733263757861, 0.05662233258418616, 0.0642949778866368, 0.06664898909402753, 0.06665370915477775, 0.07728163687154739, 0.10936508954335604, 0.14568148637618314, 0.1463533753128166, 0.14801402985877285, 0.1483425131965938, 0.14862242221387584, 0.1497865261838863, 0.16103656629569774, 0.20242646902874556, 0.2119390905411369, 0.25064403346255093, 0.28110740300030224, 0.34198798125872254, 0.4307894454387549, 0.4465975492664109, 0.45321456578786173, 0.469411951206936, 0.49194959035448055, 0.5120471526115977, 0.5170375514635317, 0.5832103371844248, 0.5916320702974267, 0.6389947280111901, 0.6592063178614546, 0.6755225964877475, 0.6945757430697759, 0.7111361873732852, 0.7158292964438517, 0.7241583474139746, 0.7365077368732541, 0.7459648221161512, 0.7722344697921394, 0.7777009272644957, 0.7933591159504617, 0.8056467687499126, 0.8240245462245174, 0.8303894582879163, 0.8282188732674562, 0.8448156808550813, 0.8603134654742013, 0.8708261456356666, 0.8838438213344135, 0.8857788042346435, 0.8895350263962168, 0.8964681587199698, 0.8989749854830673, 0.9040314304578819, 0.9121314627579208, 0.917902636398989, 0.9249188650325904, 0.931702920145623, 0.9351091821355759, 0.9405855796264918, 0.9435531123183676, 0.9445230010913208, 0.9427383593066792, 0.945555882583206, 0.9543666584729552, 0.9618759569368063, 0.9641800497510863, 0.9675345034928258, 0.9708581664595259, 0.9730670018008916, 0.9730670018008916, 0.97317075818903, 0.9762930116964027, 0.977855837924864, 0.9800867751954423, 0.9807049769247472, 0.9811002319234319, 0.9811002319234319, 0.9860348337085855, 0.9860348337085855, 0.9860348337085855, 0.9916536072765254, 0.9918621832740995, 0.9941693470323774, 0.9941693470323774, 0.9949614262402982, 0.9949614262402982, 0.9949614262402982, 0.9949614262402982, 0.9970677451971689, 0.9970677451971689, 0.9982, 0.9982, 0.9982]
weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.060892673697354426, 0.11185868939679552, 0.14887062283509414, 0.1551247378079863, 0.19951254791578585, 0.22056736687474307, 0.2216480299927513, 0.22220311343313073, 0.22220311343313073, 0.25170250463635113, 0.2912403209431512, 0.336131565705207, 0.3350716769789478, 0.3288159175087838, 0.31628936427508264, 0.3081088641725035, 0.30277078666303997, 0.3257985913891715, 0.33951748143970445, 0.3395988339342606, 0.35273026329642104, 0.3699595357023526, 0.389969489865385, 0.4035294307608058, 0.4214833550635399, 0.4303418763657443, 0.45231065581410296, 0.46625221877907463, 0.4769085314751738, 0.5307719691889645, 0.5417948365813142, 0.5532861200712721, 0.5718790421627717, 0.5820016672639565, 0.5928690619568534, 0.5992274892343541, 0.6034603267909142, 0.6572144033485725, 0.6666852032141322, 0.6769858254629307, 0.6970329227945113, 0.7049731103381014, 0.7180992312359576, 0.7296315435997485, 0.7580664114803772, 0.7956115500759324, 0.8044801392716459, 0.8141268366602785, 0.8307350406776909, 0.8544272429932281, 0.866143750210381, 0.868542889812083, 0.8858069923429449, 0.8873706553776769, 0.8919367570565024, 0.8921945841214454, 0.8921945841214454, 0.9042742522280978, 0.9042742522280978, 0.9088713685202636, 0.9166780325950553, 0.9183087008425594, 0.9235817705224088, 0.9276706520482222, 0.9316770136038312, 0.9316770136038312, 0.9359031387621253, 0.9359031387621253, 0.9382146965510699, 0.9483642920753059, 0.9542855258480916, 0.9565207056342917, 0.9619075812464366, 0.9707943913493366, 0.9707943913493366, 0.9707943913493366, 0.9707943913493366, 0.9707943913493366, 0.9729007103062072, 0.9793616553399584, 0.9793616553399584, 0.9793616553399584, 0.9793616553399584, 0.9793616553399584, 0.9793616553399584, 0.9883504193849021, 0.9883504193849021, 0.9883504193849021, 0.9883504193849021, 0.9883504193849021, 0.9883504193849021, 0.9870437956204381, 0.9870437956204381, 0.9870437956204381, 0.9870437956204381, 0.9870437956204381, 0.9870437956204381]


fig = plt.figure()
ax = plt.subplot(111)

ax.plot(labels, outlier, label="outlier")
ax.plot(labels, random, label="random")
ax.plot(labels, weights, label="weights")

ax.set_ylabel('total fscore')
ax.set_ylim((0.0, 1.0))
ax.set_xlabel('labels')

ax.legend(loc=4)

plt.show()