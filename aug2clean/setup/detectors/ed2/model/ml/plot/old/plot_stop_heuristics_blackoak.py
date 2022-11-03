import matplotlib.pyplot as plt
import numpy as np


def plot_list(ranges, list_series, list_names):
    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(len(list_series)):
        ax.plot(ranges[i], list_series[i], label=list_names[i])

    ax.set_ylabel('fscore')
    ax.set_xlabel('labels')

    ax.legend(loc=4)

    plt.show()

fscore_stop = []
fscore_stop.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20014686552596506, 0.25735613131459212, 0.41710695070700882, 0.52802367752182866, 0.59060966451567376, 0.71325111333433677, 0.71486896405770528, 0.71819397745323676, 0.73515956212987987, 0.7208068539545528, 0.72426862347828058, 0.7289810374674176, 0.7312750787597414, 0.75403878044109796, 0.7927854370760371, 0.81377281452214012, 0.85629731816700638, 0.87381608735157767, 0.87834060921743584, 0.93680882161795098, 0.93900829749751502, 0.93700817566112693, 0.93701660085943494, 0.93880241248648366, 0.94758881760774527, 0.95255696614793683, 0.96148672675829872, 0.96166709346002455, 0.96166709346002455, 0.96166709346002455, 0.96242279912406004, 0.96428663527445835, 0.96468676027631406, 0.97868676503544338, 0.98335689483147615, 0.98495072104006487, 0.98536274814140568, 0.98585372010349392, 0.98931791094146249, 0.99013606246109964])
fscore_stop.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19812920063745756, 0.2394658339819935, 0.40410093351269827, 0.54217746562028302, 0.64821193584255099, 0.76828710985026083, 0.76958123059365813, 0.78244517204446418, 0.79490072362268316, 0.79011269170749598, 0.79555303453911996, 0.80017561195665021, 0.80108771661385114, 0.80223384713788326, 0.80383664860774573, 0.81450870881987658, 0.86466967414951956, 0.87154627687298925, 0.8794850701067648, 0.92494208782391496, 0.92636592219298286, 0.92928794213285559, 0.92940587870736924, 0.92934236364577838, 0.94035997559487494, 0.95182933522914392, 0.94427255207370842, 0.944079441505973, 0.944079441505973, 0.944079441505973, 0.94521637032785932, 0.94838319313613406, 0.95204519610103333, 0.95915937153063957, 0.97090506883147332, 0.97183655141445879, 0.96944246531105993, 0.97652580945818956, 0.97761017416033302, 0.97932582173250549])
fscore_stop.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19529642244129516, 0.25599867192787545, 0.40875391076849027, 0.51525558435062924, 0.59531298440102132, 0.72277815132697865, 0.72567790256056453, 0.74048270380611425, 0.75831116538978438, 0.78097016754285975, 0.79839729963024519, 0.79526114194228503, 0.79744808212344775, 0.79837755085936157, 0.82326075326769987, 0.8318843593167522, 0.8834112776676577, 0.88819357906579066, 0.8901613098660176, 0.90739000561718119, 0.90757264113518266, 0.91053384082051014, 0.91013718149635214, 0.91125372018949125, 0.92923202239779923, 0.9360185707367018, 0.93612314427983412, 0.93649867496729189, 0.93649867496729189, 0.93649867496729189, 0.93758845902449328, 0.9400970871294092, 0.94059054996763691, 0.96868342789647965, 0.97493610842843148, 0.97314799878821734, 0.96955935181263442, 0.97420630015315124, 0.97504618910638707, 0.97930247909351187])
fscore_stop.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20147373092149051, 0.24128830770354676, 0.40673739065698866, 0.45613835648621814, 0.57019096493407462, 0.69276850275836266, 0.69367358878628693, 0.67701977276698866, 0.69313730067442825, 0.68209606171758008, 0.68462259020165661, 0.69690455831782483, 0.69849276120912196, 0.74302659338004962, 0.75140626942361077, 0.76365482570221077, 0.80692534991649756, 0.84270461737855717, 0.84874846951011695, 0.90335948480479578, 0.90336624386405606, 0.90180613347656424, 0.9019477888512486, 0.92462753018707422, 0.94049318420681105, 0.95245753633068619, 0.96309009072404628, 0.96319855912318897, 0.96319855912318897, 0.96319855912318897, 0.96514869525606817, 0.96514869525606817, 0.96514152146638843, 0.97624109478299059, 0.98034124436072623, 0.96803011330743982, 0.96802212528737241, 0.96812359914425095, 0.96779041290064582, 0.97106852983869318])
fscore_stop.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19492157991294651, 0.25003575468386358, 0.4035828954195585, 0.50859514336671696, 0.62304759369592966, 0.74166689649466333, 0.7465360814829568, 0.76385204993017153, 0.77696297526149749, 0.79652258495957196, 0.8130543437155211, 0.81582870696171195, 0.82179470355108786, 0.85286789114431572, 0.85987687489850695, 0.86563270191315844, 0.89820855417294443, 0.90001426838838561, 0.90657361663865288, 0.92624059124945601, 0.9274859916267616, 0.93054975247620819, 0.9313727771185718, 0.93211218048225186, 0.93975951400411173, 0.94754760291863904, 0.95845773671385093, 0.95857883697153046, 0.95857883697153046, 0.95857883697153046, 0.95765402036195812, 0.95768572857180456, 0.9644645609875111, 0.97315579588402756, 0.97034484441607383, 0.97034484441607383, 0.97100216179256549, 0.97715813094833603, 0.97845785339027547, 0.98702340857377735])

average_stop = list(np.mean(np.matrix(fscore_stop), axis=0).A1)
label_stop = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440]


fscore_full = []
fscore_full.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20089419053991789, 0.26465548757658647, 0.42265622814484455, 0.52618037786112382, 0.6386219488756486, 0.75733679138966437, 0.75927332236722278, 0.77546596596183026, 0.74135392156357971, 0.7480183512392401, 0.7523891528997132, 0.74342952799420414, 0.74514657336086354, 0.77364803382190306, 0.7871052796785738, 0.79421066105787319, 0.84467853556762096, 0.84764606304616363, 0.90295971602895297, 0.94031776439583681, 0.94102887793532974, 0.95310691602432818, 0.95324920425141368, 0.95098265655939151, 0.95178133942725396, 0.95877053524112343, 0.95877322689718636, 0.95894156343751791, 0.95907066795740559, 0.95917545571404239, 0.9592904088513029, 0.95789183734412497, 0.95789183734412497, 0.95996845057106273, 0.97061436355654251, 0.97603776182719859, 0.97124425810031756, 0.9717918299515319, 0.97179307898349188, 0.97179307898349188, 0.97213293492627595, 0.97323817100896259, 0.97324213645802349, 0.97348210366987709, 0.97685663869429562, 0.9828576621252344, 0.99011326047816439, 0.99010136775152535, 0.99010136775152535, 0.99010136775152535])
fscore_full.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.17959813640012118, 0.21792368817646254, 0.37390394131960586, 0.44101162426484353, 0.49330512331877818, 0.63610413970216584, 0.63705565602999936, 0.65699545810043847, 0.67759009786826752, 0.68371809602023648, 0.71384281612367351, 0.71655634595841688, 0.71934604157816118, 0.75214153606541412, 0.80322343095796489, 0.81425405908437609, 0.86777188852376363, 0.86996053774644799, 0.87192039724883508, 0.91387669064957111, 0.91437396719807928, 0.91557720152691857, 0.91558978409689307, 0.94593370225694451, 0.94289931062327625, 0.94846778780742536, 0.94846778780742536, 0.94865930780263574, 0.9486564707310986, 0.9486564707310986, 0.94950329784303233, 0.95413852256888598, 0.9542856759998124, 0.95436563243776396, 0.97283390541845405, 0.97860211316101875, 0.97516599244120017, 0.97503262515007572, 0.97503530032024766, 0.97503530032024766, 0.97538941948690105, 0.97524493931354317, 0.97524895320189009, 0.97526295998412305, 0.98001885978315184, 0.98021564098831304, 0.98604610748407329, 0.98674113564311738, 0.98674113564311738, 0.98674113564311738])
fscore_full.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.19878082210875028, 0.24150070794501738, 0.35317078067359425, 0.46073630078670436, 0.51662840785566522, 0.65532286212914481, 0.65992326867538242, 0.68099649348989444, 0.65973170936757819, 0.68910295471992533, 0.69369666420319243, 0.70380867736116559, 0.7175265130372378, 0.74113996448597663, 0.7769814450036483, 0.77879129031498995, 0.80579109550079031, 0.80623652170549387, 0.85087531261164706, 0.86408623473032586, 0.86455819886991014, 0.86631928309808781, 0.90745044174525968, 0.91360008785569169, 0.92702918779366417, 0.9380522648317754, 0.94916312599477604, 0.94652322587673277, 0.94652322587673277, 0.94652322587673277, 0.94596061826797362, 0.9462184606235664, 0.94706583839726421, 0.94651250147665678, 0.95682724403517205, 0.95922002656989436, 0.97120514196651397, 0.97377431545994719, 0.97377431545994719, 0.97377431545994719, 0.97426466326928796, 0.97426525983935586, 0.97427582751990638, 0.9747445492249589, 0.97762949611492289, 0.98162849424172816, 0.97010756886206151, 0.9706463171627453, 0.9706463171627453, 0.9706463171627453])
fscore_full.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1694274784585634, 0.21041241508560149, 0.37304752017365861, 0.47816380274860742, 0.56150210551760116, 0.68928167806893714, 0.69186215495562298, 0.71085268424319592, 0.72968743632584854, 0.73422638416699371, 0.76994812905406829, 0.77916768095570899, 0.77967557039583124, 0.81343988790589306, 0.84334077331230539, 0.85610196965011243, 0.89920966622224563, 0.9006444958718397, 0.90279005013973368, 0.94288415387971791, 0.94137224209407977, 0.94409472501147662, 0.94416715031921072, 0.94606810075100056, 0.95389116945480512, 0.95905884693566357, 0.94897367859212489, 0.94894284430841391, 0.94894284430841391, 0.94894284430841391, 0.95138163045761526, 0.9514945039835615, 0.95150117825592662, 0.95160351339817462, 0.96281176880569519, 0.96607762984225432, 0.97991387013405928, 0.98069813698230712, 0.98069813698230712, 0.98069813698230712, 0.98103509301575076, 0.98104967839236512, 0.98105100431498904, 0.98126850642423369, 0.98335704828224868, 0.98554513321572945, 0.99210201153938782, 0.991959041366963, 0.991959041366963, 0.991959041366963])
fscore_full.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20044418991213978, 0.2314610904151316, 0.39971751412429374, 0.47197700334594628, 0.55417076877977045, 0.69004704966746855, 0.69324375973892327, 0.71234397633731472, 0.73161883675381079, 0.71372237272888928, 0.7161053085253688, 0.72421301787433723, 0.71938374100212554, 0.74308439090005263, 0.77206693016732542, 0.78333618322453447, 0.82903145037503134, 0.83107001410491843, 0.83325864578968145, 0.89329845559551968, 0.89456234179022098, 0.8986342274563881, 0.9028489902592689, 0.9240612162352505, 0.93395877471693711, 0.94074777784484409, 0.94635762108392341, 0.94652151114102567, 0.9466053291630091, 0.9466053291630091, 0.947241890119356, 0.95004944631849686, 0.95062096554705777, 0.96250357614182747, 0.96644376996213321, 0.96289921465106076, 0.96255433053474659, 0.9624886453714524, 0.9624886453714524, 0.9624886453714524, 0.9626441995429792, 0.96263228208958362, 0.96283957888537142, 0.96274394783153094, 0.97385101575461419, 0.97818008352581365, 0.98014647260109178, 0.98065266669963225, 0.98065134024830936, 0.98065134024830936])

average_full = list(np.mean(np.matrix(fscore_full), axis=0).A1)
label_full = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540]


ranges = [label_stop, label_full]
list = [average_stop, average_full]
names = ["stop heuristic", "full round robin"]

plot_list(ranges, list, names)