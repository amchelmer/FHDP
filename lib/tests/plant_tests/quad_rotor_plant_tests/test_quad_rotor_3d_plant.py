import numpy as np

from ..plant_base_test import PlantBaseTest
from ....controllers import FixedInputController
from ....features import Feature
from ....plants.quad_rotor_plant import QuadRotor3DPlant
from ....sets import FeatureSet

from numpy.testing import assert_array_almost_equal


class TestQuadRotor3DPlant(PlantBaseTest):
    np.seterr(divide="raise", invalid="raise")
    SEED = 1804
    TIME_STEP = 0.005
    MOD = []
    STATE = np.array([[
        0.4, 0.2, -0.1,
        1.1, 1.2, -1.3,
        0.662146043378, 0.1891370141, 0.0945685087871, 0.718920443639,
        0.05, -0.013, 0.098,
    ]]).T
    PREPROCESSED_STATE = STATE.copy()
    ACTION = np.array([[
        0.1,
        -0.1,
        0.08,
        -0.05
    ]]).T
    PREPROCESSED_ACTION = np.array([[
        2262.36143631,
        2144.1652219,
        2025.96900749,
        1848.67468588]]).T
    ROTOR_SPEEDS = PREPROCESSED_ACTION.copy()
    POWER = np.array([[
        6.137080747782433,
        5.224570720095714,
        4.407316727777153,
        3.348554354140658,
    ]]).T
    INPUT_VOLTAGE = np.array([[
        2,
        2.5,
        1.5,
        1.2
    ]]).T
    OLD_ROTOR_SPEEDS = np.array([[
        2.367956921155402e3,
        2.236861499020690e3,
        2.124085042403219e3,
        1.940265185068506e3,
    ]]).T
    THRUST = np.array([[
        1.425869821719887,
        1.280773678806434,
        1.143461360273533,
        0.952087553192865,
    ]]).T
    VELOCITIES = np.array([
        [0.937936189761434, -1.636584491816809, -0.882282398280747],
        [0.937936189761434, -1.636584491816809, -0.876732398280747],
        [0.937936189761434, -1.636584491816809, -0.886182398280747],
        [0.937936189761434, -1.636584491816809, -0.891732398280747],
    ])
    THRUST_RATIOS = np.array([[
        0.963839594317401,
        0.964099348356216,
        0.963657131013367,
        0.963397566407162,
    ]]).T
    PROPELLER_FORCES_MOMENTS = np.array([[
        -0.018941131873420,
        0.033049781101925,
        -4.628246385263388,
        -0.047347497727785,
        0.041024465244362,
        -0.000320063580120,
    ]]).T
    FUSELAGE_FORCES_MOMENTS = np.array([[
        -0.071810302880146,
        0.125300238256301,
        0.077824994174961,
        0,
        0,
        -0.000029412250000,
    ]]).T
    BODY_VELOCITY_VECTOR = np.array([[
        +0.937936189761434,
        -1.636584491816809,
        -0.884232398280747,
    ]]).T
    NEXT_STATE = np.array([[
        0.4055, 0.206, -0.1065,
        1.076812219465095, 1.204936941166109, -1.299723755776767,
        0.661949309343074, 0.189266304878482, 0.094590512956160, 0.719064678979967,
        -0.055210291617301, 0.078189978320805, 0.097611693522089
    ]]).T
    DERIVATIVE = np.array([[
        1.1, 1.2, -1.3,
        -4.637556106981045, 0.987388233221743, 0.055248844646686,
        -0.039340831985000, 0.025860490915000, 0.004401348500000, 0.028851552645000,
        -21.042058323460186, 18.237995664161065, -0.077661295582294,
    ]]).T
    INITIAL_STATE = np.array([[
        0, 0, 0,
        0, 0, 0,
        1., 0, 0, 0,
        0, 0, 0,
    ]]).T
    LIKE_ME_ARRAY = np.array([[0.1, -0.2, 0.3, 0.4]]).T
    LIKE_ME_OTHER_FEATURE_SET = FeatureSet([
        Feature(r"$x$ [m]"),
        Feature(r"$\dot{x}$ [m/s]", scale=10.),
        Feature(r"$\dot{\theta}$ [-]", scale=10.),
        Feature(r"$\u_1$ [-]", feature_type="action", bounds=np.array([-1, 1])),
    ])
    OUT_OF_BOUNDS_STATE = np.array([[
        0.4, 0.2, -25.1,
        1.1, 1.2, -1.3,
        0.662146043378, 0.1891370141, 0.0945685087871, 0.718920443639,
        0.05, -0.013, 0.098,
    ]]).T
    VALIDATION_LENGTH = 0.5
    VALIDATION_XYZ = np.array([
        [0.4, 0.2, -0.1],
        [0.4055, 0.206, -0.1065],
        [0.410884061097, 0.212024684706, -0.112998618779],
        [0.416152293318, 0.218073987293, -0.11949564329],
        [0.421304946655, 0.224147728931, -0.125990928659],
        [0.426342411659, 0.23024562057, -0.13248439641],
        [0.431265217366, 0.236367264072, -0.13897603525],
        [0.436074030334, 0.242512153517, -0.145465900495],
        [0.440769654097, 0.248679676538, -0.151954112903],
        [0.445353028762, 0.254869115643, -0.158440857181],
        [0.449825230716, 0.261079649501, -0.164926380188],
        [0.454187472472, 0.267310354219, -0.171410988835],
        [0.458441102616, 0.273560204594, -0.177895047686],
        [0.462587605865, 0.279828075347, -0.18437897626],
        [0.466628603226, 0.28611274235, -0.190863246049],
        [0.470565852233, 0.29241288384, -0.197348377239],
        [0.474401247275, 0.298727081621, -0.203834935152],
        [0.47813681998, 0.305053822274, -0.210323526396],
        [0.481774739669, 0.311391498356, -0.216814794733],
        [0.485317313846, 0.31773840961, -0.223309416665],
        [0.488766988729, 0.324092764184, -0.229808096732],
        [0.492126349808, 0.330452679862, -0.236311562528],
        [0.495398122099, 0.336816185118, -0.242820559778],
        [0.498585170076, 0.343181219913, -0.249335847558],
        [0.501690499282, 0.349545637647, -0.25585819125],
        [0.504717256852, 0.355907206516, -0.262388356401],
        [0.507668731995, 0.3622636109, -0.268927102305],
        [0.510548356416, 0.368612452812, -0.275475175285],
        [0.51335970466, 0.374951253405, -0.282033301688],
        [0.51610649437, 0.381277454543, -0.28860218059],
        [0.518792586438, 0.387588420455, -0.29518247621],
        [0.521421985026, 0.393881439478, -0.301774810029],
        [0.523998837448, 0.400153725893, -0.308379752629],
        [0.526527433886, 0.406402421876, -0.314997815241],
        [0.52901220692, 0.412624599572, -0.321629441014],
        [0.531457730858, 0.418817263301, -0.328274996004],
        [0.533868720827, 0.424977351912, -0.3349347599],
        [0.536250031628, 0.431101741307, -0.341608916477],
        [0.538606656293, 0.437187247132, -0.348297543809],
        [0.540943724361, 0.443230627669, -0.355000604233],
        [0.543266499809, 0.449228586932, -0.361717934091],
        [0.545580378632, 0.455177777984, -0.36844923326],
        [0.547890886041, 0.461074806503, -0.375194054495],
        [0.550203673243, 0.466916234593, -0.381951792598],
        [0.552524513776, 0.472698584874, -0.388721673448],
        [0.55485929938, 0.478418344855, -0.395502742908],
        [0.557214035351, 0.484071971616, -0.402293855654],
        [0.559594835376, 0.489655896802, -0.409093663943],
        [0.562007915791, 0.495166531958, -0.415900606377],
        [0.564459589261, 0.500600274197, -0.422712896684],
        [0.566956257826, 0.505953512239, -0.429528512581],
        [0.569504405307, 0.511222632812, -0.436345184749],
        [0.572110589035, 0.516404027435, -0.443160385988],
        [0.574781430877, 0.52149409958, -0.449971320594],
        [0.577523607548, 0.526489272234, -0.456774914031],
        [0.580343840178, 0.531385995844, -0.463567802949],
        [0.583248883117, 0.53618075667, -0.470346325622],
        [0.586245511981, 0.540870085509, -0.477106512873],
        [0.589340510898, 0.545450566826, -0.48384407956],
        [0.592540658985, 0.549918848249, -0.490554416697],
        [0.595852716021, 0.554271650424, -0.497232584285],
        [0.599283407346, 0.558505777234, -0.503873304944],
        [0.602839407975, 0.562618126326, -0.510470958407],
        [0.606527325954, 0.566605699957, -0.517019576983],
        [0.610353684974, 0.570465616106, -0.523512842051],
        [0.614324906272, 0.574195119835, -0.529944081687],
        [0.618447289847, 0.577791594852, -0.536306269482],
        [0.622726995044, 0.581252575245, -0.542592024666],
        [0.627170020539, 0.584575757337, -0.548793613588],
        [0.631782183801, 0.587759011614, -0.554902952647],
        [0.636569100074, 0.590800394677, -0.56091161274],
        [0.641536160972, 0.593698161164, -0.566810825304],
        [0.646688512752, 0.596450775565, -0.572591490013],
        [0.652031034374, 0.599056923905, -0.578244184189],
        [0.657568315426, 0.60151552518, -0.583759173985],
        [0.663304634049, 0.603825742526, -0.589126427369],
        [0.66924393495, 0.605986994006, -0.594335628967],
        [0.675389807662, 0.607998962978, -0.599376196766],
        [0.681745465156, 0.60986160794, -0.604237300694],
        [0.688313722974, 0.611575171789, -0.608907883085],
        [0.695096979448, 0.613140189445, -0.613376680719],
        [0.702097196497, 0.614557495179, -0.617632248646],
        [0.709315880527, 0.615828230998, -0.621662986352],
        [0.716754065108, 0.616953852756, -0.625457165319],
        [0.724412294928, 0.617936135291, -0.629002958173],
        [0.7322906112, 0.618777176495, -0.6322884693],
        [0.740388538648, 0.619479400269, -0.635301766811],
        [0.748705074241, 0.620045558282, -0.638030915669],
        [0.757238678112, 0.62047872943, -0.640464011036],
        [0.765987266154, 0.620782318633, -0.642589212623],
        [0.774948204098, 0.620960056393, -0.644394781322],
        [0.784118304294, 0.621015995417, -0.645869115145],
        [0.793493824707, 0.620954505893, -0.647000785321],
        [0.803070470181, 0.620780269423, -0.647778572315],
        [0.812843395994, 0.62049827136, -0.64819150118],
        [0.822807213636, 0.620113791369, -0.648228875506],
        [0.832955998666, 0.61963239492, -0.647880313547],
        [0.84328330096, 0.619059922233, -0.647135781761],
        [0.853782157062, 0.618402476099, -0.64598562722],
        [0.864445104554, 0.617666408674, -0.644420608686],
    ]).T
    VALIDATION_ATTITUDE = np.array([
        [0.66214604, 0.18913701, 0.09456851, 0.71892045],
        [0.661949309343, 0.189266304878, 0.094590512956, 0.71906467898],
        [0.661781431433, 0.189057451522, 0.094574466, 0.719276224698],
        [0.661642138378, 0.188512497419, 0.094520187259, 0.719554482468],
        [0.661530927654, 0.187633413939, 0.09442746645, 0.719898595158],
        [0.66144706577, 0.186422097541, 0.094296065306, 0.720307457132],
        [0.661389592924, 0.184880371868, 0.094125718232, 0.720779719218],
        [0.661357327555, 0.183009990375, 0.093916133012, 0.721313793485],
        [0.66134887078, 0.180812639495, 0.093666991567, 0.721907857838],
        [0.661362610725, 0.178289942297, 0.093377950779, 0.72255986044],
        [0.661396726757, 0.175443462625, 0.093048643358, 0.723267523968],
        [0.661449193606, 0.172274709701, 0.092678678771, 0.724028349705],
        [0.661517785406, 0.168785143169, 0.09226764422, 0.72483962148],
        [0.661600079642, 0.164976178568, 0.09181510567, 0.72569840946],
        [0.661693461013, 0.160849193207, 0.091320608934, 0.726601573822],
        [0.66179512523, 0.156405532439, 0.090783680806, 0.727545768281],
        [0.661902082739, 0.151646516307, 0.090203830249, 0.728527443523],
        [0.662011162396, 0.146573446553, 0.089580549627, 0.729542850528],
        [0.662119015093, 0.141187613977, 0.088913316, 0.730588043805],
        [0.662222117339, 0.135490306123, 0.088201592458, 0.731658884551],
        [0.662316774825, 0.129482815291, 0.08744482952, 0.732751043752],
        [0.662399125968, 0.123166446851, 0.086642466568, 0.733860005228],
        [0.662465145449, 0.116542527855, 0.08579393335, 0.734981068645],
        [0.662510647759, 0.109612415929, 0.084898651526, 0.736109352508],
        [0.662531290766, 0.102377508438, 0.083956036265, 0.737239797146],
        [0.662522579315, 0.094839251902, 0.082965497904, 0.738367167712],
        [0.662479868864, 0.08699915166, 0.081926443649, 0.739486057198],
        [0.662398369193, 0.078858781764, 0.080838279339, 0.740590889508],
        [0.662273148166, 0.070419795101, 0.079700411255, 0.741675922571],
        [0.662099135595, 0.061683933709, 0.078512247992, 0.742735251541],
        [0.661871127192, 0.052653039306, 0.077273202378, 0.743762812081],
        [0.661583788637, 0.043329063991, 0.075982693451, 0.744752383764],
        [0.661231659773, 0.033714081119, 0.074640148487, 0.745697593587],
        [0.660809158945, 0.023810296332, 0.073245005088, 0.74659191964],
        [0.660310587493, 0.013620058728, 0.071796713319, 0.747428694928],
        [0.659730134412, 0.003145872164, 0.070294737898, 0.748201111374],
        [0.659061881204, -0.007609593335, 0.068738560449, 0.748902224019],
        [0.658299806926, -0.018643490076, 0.067127681793, 0.749524955433],
        [0.657437793449, -0.029952781107, 0.065461624303, 0.750062100357],
        [0.65646963095, -0.041534228587, 0.063739934304, 0.7505063306],
        [0.655389023641, -0.053384382098, 0.061962184529, 0.750850200191],
        [0.654189595758, -0.06549956695, 0.060127976612, 0.751086150824],
        [0.652864897813, -0.077875872482, 0.058236943642, 0.751206517599],
        [0.651408413135, -0.090509140391, 0.056288752751, 0.75120353508],
        [0.649813564695, -0.103394953111, 0.054283107751, 0.751069343685],
        [0.648073722247, -0.116528622264, 0.052219751807, 0.750795996425],
        [0.64618220978, -0.129905177213, 0.05009847015, 0.750375466008],
        [0.644132313296, -0.143519353744, 0.047919092821, 0.749799652315],
        [0.641917288932, -0.157365582908, 0.045681497452, 0.749060390274],
        [0.639530371416, -0.171437980049, 0.043385612065, 0.74814945813],
        [0.636964782887, -0.185730334051, 0.0410314179, 0.747058586136],
        [0.634213742074, -0.200236096849, 0.038618952258, 0.745779465667],
        [0.631270473827, -0.214948373212, 0.036148311358, 0.744303758766],
        [0.628128219037, -0.229859910868, 0.033619653195, 0.74262310814],
        [0.624780244912, -0.244963090977, 0.031033200414, 0.740729147596],
        [0.621219855634, -0.260249919009, 0.028389243161, 0.738613512938],
        [0.617440403387, -0.275712016058, 0.025688141934, 0.736267853318],
        [0.613435299754, -0.291340610638, 0.022930330415, 0.733683843052],
        [0.609198027485, -0.307126530991, 0.020116318268, 0.730853193883],
        [0.604722152623, -0.323060197959, 0.017246693912, 0.727767667714],
        [0.600001336985, -0.339131618461, 0.014322127248, 0.724419089787],
        [0.595029350987, -0.355330379603, 0.01134337233, 0.720799362305],
        [0.589800086808, -0.371645643482, 0.00831126999, 0.7169004785],
        [0.584307571861, -0.388066142717, 0.005226750377, 0.712714537121],
        [0.578545982582, -0.404580176741, 0.002090835435, 0.708233757338],
        [0.572509658498, -0.421175608913, -0.001095358723, 0.703450494045],
        [0.566193116561, -0.437839864469, -0.004330619527, 0.698357253542],
        [0.559591065734, -0.454559929371, -0.007613635772, 0.692946709575],
        [0.552698421788, -0.471322350073, -0.010942995563, 0.687211719722],
        [0.545510322302, -0.48811323426, -0.014317184381, 0.681145342077],
        [0.53802214183, -0.50491825257, -0.017734583293, 0.674740852237],
        [0.530229507199, -0.521722641357, -0.021193467302, 0.667991760529],
        [0.522128312919, -0.538511206507, -0.024692003864, 0.660891829469],
        [0.513714736667, -0.555268328335, -0.028228251566, 0.653435091415],
        [0.504985254802, -0.5719779676, -0.031800158979, 0.645615866365],
        [0.495936657889, -0.588623672646, -0.035405563708, 0.637428779882],
        [0.486566066186, -0.605188587693, -0.039042191622, 0.628868781095],
        [0.47687094506, -0.621655462293, -0.042707656298, 0.619931160737],
        [0.466849120281, -0.638006661958, -0.04639945867, 0.610611569187],
        [0.456498793179, -0.654224179978, -0.050114986897, 0.600906034457],
        [0.4458185556, -0.670289650413, -0.053851516461, 0.590810980101],
        [0.434807404628, -0.686184362283, -0.057606210484, 0.580323242989],
        [0.423464757046, -0.701889274932, -0.061376120291, 0.569440090908],
        [0.411790463472, -0.717385034574, -0.065158186208, 0.558159239942],
        [0.399784822157, -0.732651991995, -0.06894923861, 0.546478871589],
        [0.387448592378, -0.747670221415, -0.072745999207, 0.534397649576],
        [0.374783007418, -0.76241954047, -0.076545082587, 0.521914736323],
        [0.361789787076, -0.776879531308, -0.080342998009, 0.509029809022],
        [0.348471149678, -0.791029562768, -0.084136151441, 0.495743075282],
        [0.334829823561, -0.804848813614, -0.087920847856, 0.482055288314],
        [0.320869057989, -0.818316296795, -0.091693293776, 0.467967761605],
        [0.306592633484, -0.831410884699, -0.095449600063, 0.453482383059],
        [0.292004871524, -0.844111335367, -0.099185784955, 0.438601628558],
        [0.277110643602, -0.85639631963, -0.102897777351, 0.423328574918],
        [0.261915379617, -0.868244449128, -0.106581420329, 0.40766691222],
        [0.246425075562, -0.87963430518, -0.110232474901, 0.391620955467],
        [0.230646300515, -0.890544468461, -0.113846624001, 0.375195655566],
        [0.214586202889, -0.900953549443, -0.117419476696, 0.358396609593],
        [0.198252515942, -0.910840219576, -0.120946572616, 0.34123007033],
        [0.181653562529, -0.920183243153, -0.124423386607, 0.323702955048],
    ]).T
    SIMULATION_INITIAL_STATE = STATE
    SIMULATION_LENGTH = 5 * TIME_STEP
    SIMULATION_CONTROLLER = FixedInputController(ACTION)
    SIMULATION_STATES = np.array([
        [0.4, 0.4055, 0.41088406, 0.41615229, 0.42130495],
        [0.2, 0.206, 0.21202468, 0.21807399, 0.22414773],
        [-0.1, -0.1065, -0.11299862, -0.11949564, -0.12599093],
        [1.1, 1.07681222, 1.05364644, 1.03053067, 1.007493],
        [1.2, 1.20493694, 1.20986052, 1.21474833, 1.21957833],
        [-1.3, -1.29972376, -1.2994049, -1.29905707, -1.29869355],
        [0.66214604, 0.66194931, 0.66178144, 0.66164214, 0.66153093],
        [0.18913701, 0.18926631, 0.18905746, 0.1885125, 0.18763342],
        [0.09456851, 0.09459051, 0.09457446, 0.09452019, 0.09442747],
        [0.71892044, 0.71906467, 0.71927622, 0.71955448, 0.71989859],
        [0.05, -0.05521029, -0.15986547, -0.26396705, -0.36751681],
        [-0.013, 0.07818998, 0.16873289, 0.25863278, 0.34789392],
        [0.098, 0.09761169, 0.09722365, 0.09683585, 0.09644832]
    ])
    SIMULATION_ACTIONS = np.tile(ACTION, (1, 5))

    @staticmethod
    def _get_plant_cls():
        return QuadRotor3DPlant

    def _get_plant_parameters(self):
        return {
            "dt": self.TIME_STEP,
            "integrator": "euler",
            "blade_flapping": True,
        }

    def _get_other_plant_parameters(self):
        return {
            "dt": self.TIME_STEP + 0.01,
            "integrator": "euler",
            "blade_flapping": True,
        }

    def plant_base_test(self):
        self._plant_base_test()

    def test_compute_thrust(self):
        plant = self._generate_plant()
        assert_array_almost_equal(
            plant._compute_thrust(self.POWER),
            self.THRUST,
            decimal=10
        )

    def test_compute_total_velocity(self):
        plant = self._generate_plant()
        _, translational_velocities, attitude_euler, omega_body = plant.split_state(self.PREPROCESSED_STATE)
        velocities_body = plant.rotate_earth_to_body(attitude_euler, translational_velocities)
        assert_array_almost_equal(
            plant._compute_total_velocity(velocities_body, omega_body),
            self.VELOCITIES
        )

    def test_compute_thrust_ratio(self):
        plant_cls = self._get_plant_cls()
        assert_array_almost_equal(
            plant_cls._compute_thrust_ratio(self.VELOCITIES),
            self.THRUST_RATIOS
        )

    def test_get_power_by_voltage(self):
        plant = self._generate_plant()
        new_rotor_speeds, power = plant.get_power_by_voltage(
            self.OLD_ROTOR_SPEEDS,
            self.INPUT_VOLTAGE,
        )
        assert_array_almost_equal(
            new_rotor_speeds,
            self.ROTOR_SPEEDS,
            decimal=10
        )
        assert_array_almost_equal(
            power,
            self.POWER,
            decimal=10
        )

    def test_get_power_by_omega(self):
        plant = self._generate_plant()
        assert_array_almost_equal(
            plant.get_power_by_omega(
                self.ROTOR_SPEEDS
            ),
            self.POWER
        )

    def test_get_lift_coefficient(self):
        plant = self._generate_plant()
        assert_array_almost_equal(
            np.array([plant.get_lift_coefficient(alpha) for alpha in [0.3840, 0.7854, 1.3701]]),
            np.array([
                0.400015793330236,
                0.645000526147874,
                0.259981049782518
            ]),
            decimal=10
        )

    def test_split_state(self):
        plant = self._generate_plant()
        for x, y in zip(
                plant.split_state(self.STATE),
                (self.STATE[:3], self.STATE[3:6], self.STATE[6:10], self.STATE[10:])
        ):
            assert_array_almost_equal(x, y, decimal=10)

    def test_get_forces_and_moments(self):
        plant = self._generate_plant()
        forces_moments = plant._get_propeller_forces_and_moments(
            self.PREPROCESSED_STATE,
            self.PREPROCESSED_ACTION
        ) + plant._get_fuselage_forces_and_moments(
            self.PREPROCESSED_STATE
        )
        assert_array_almost_equal(
            plant.get_forces_and_moments(
                self.PREPROCESSED_STATE,
                self.PREPROCESSED_ACTION
            )[0],
            forces_moments[:3],
            decimal=10
        )
        assert_array_almost_equal(
            plant.get_forces_and_moments(
                self.PREPROCESSED_STATE,
                self.PREPROCESSED_ACTION
            )[1],
            forces_moments[3:],
            decimal=10
        )

    def test_get_propeller_forces_and_moments(self):
        plant = self._generate_plant()
        assert_array_almost_equal(
            plant._get_propeller_forces_and_moments(
                self.PREPROCESSED_STATE,
                self.PREPROCESSED_ACTION
            ),
            self.PROPELLER_FORCES_MOMENTS,
            decimal=8
        )

    def test_get_fuselage_forces_and_moments(self):
        plant = self._generate_plant()
        assert_array_almost_equal(
            plant._get_fuselage_forces_and_moments(
                self.PREPROCESSED_STATE
            ),
            self.FUSELAGE_FORCES_MOMENTS,
            decimal=8
        )

    def test_rotate_earth_to_body(self):
        plant_cls = self._get_plant_cls()
        position, velocity, attitude, omega_body = plant_cls.split_state(
            self.PREPROCESSED_STATE
        )
        assert_array_almost_equal(
            plant_cls.rotate_earth_to_body(
                attitude,
                velocity,
            ),
            self.BODY_VELOCITY_VECTOR,
            decimal=8
        )

    def test_rotate_body_to_earth(self):
        plant_cls = self._get_plant_cls()
        position, velocity, attitude, omega_body = plant_cls.split_state(
            self.PREPROCESSED_STATE
        )
        assert_array_almost_equal(
            plant_cls.rotate_body_to_earth(
                attitude,
                self.BODY_VELOCITY_VECTOR,
            ),
            velocity,
            decimal=6
        )

    def test_map_input(self):
        plant_cls = self._get_plant_cls()
        assert_array_almost_equal(
            plant_cls.map_input(
                np.array([[
                    +114.26576087,
                    -0.319836956522,
                    -6.56929347826,
                    -0.510054347826,
                ]]).T
            ),
            np.array([[
                108.2065217391304,
                114.0755434782609,
                121.3451086956522,
                113.4358695652174,
            ]]).T
        )

    def test_preprocess_state_action(self):
        plant = self._generate_plant()
        preprocessed_state, preprocessed_action = plant.preprocess_state_action(self.STATE, self.ACTION)
        assert_array_almost_equal(
            preprocessed_state,
            self.PREPROCESSED_STATE
        )
        assert_array_almost_equal(
            preprocessed_action,
            self.PREPROCESSED_ACTION
        )
        assert_array_almost_equal(
            plant.preprocess_state_action(
                self.STATE,
                np.array([[2e5, 0, 0, 0]]).T
            )[1],
            plant._ROTOR_SPEED_SATURATION[:, 1:]
        )

    def test_postprocess_next_state(self):
        plant = self._generate_plant()
        assert_array_almost_equal(
            plant.postprocess_next_state(self.NEXT_STATE),
            self.NEXT_STATE,
        )

    def test_get_wb(self):
        plant_cls = self._get_plant_cls()
        quaternions = np.array([[
            0.662146043378,
            0.1891370141,
            0.0945685087871,
            0.718920443639,
        ]]).T
        assert_array_almost_equal(
            plant_cls.get_wb(quaternions),
            np.array([
                [-0.1891370141, 0.662146043378, 0.718920443639, -0.0945685087871],
                [-0.0945685087871, -0.718920443639, 0.662146043378, 0.1891370141],
                [-0.718920443639, 0.0945685087871, -0.1891370141, 0.662146043378],
            ]),
            decimal=8
        )

    def test_validate_plant_dynamics(self):
        plant = self._generate_plant()
        sim = plant.simulate(self.VALIDATION_LENGTH, self.SIMULATION_CONTROLLER, initial_state=self.STATE)
        states = sim.get_states()
        assert_array_almost_equal(
            states[0:3, :],
            self.VALIDATION_XYZ,
            decimal=8
        )
        assert_array_almost_equal(
            states[6:10, :],
            self.VALIDATION_ATTITUDE,
            decimal=8
        )
        assert_array_almost_equal(
            sim.get_actions(),
            self.ACTION.repeat(states.shape[1], axis=1),
            decimal=8
        )