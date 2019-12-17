extern crate rand;
extern crate regex;

use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::time::SystemTime;

use rand::Rng;
use regex::Regex;

#[derive(Debug)]
pub struct HmmModel {
    pub state_transition_matrix: Box<[Box<[f64]>]>,
    pub observation_probability_matrix: Box<[Box<[f64]>]>,
    pub initial_state_distribution_vector: Box<[f64]>,
    pub log_probability: f64,
    alpha_matrix: Box<[Box<[f64]>]>,
    beta_matrix: Box<[Box<[f64]>]>,
    gamma_matrix: Box<[Box<[f64]>]>,
    di_gamma_tensor: Box<[Box<[Box<[f64]>]>]>,
    scale_factors: Box<[f64]>,
}

impl HmmModel {
    pub fn train_model(number_of_hidden_state_symbols: usize, number_of_observation_symbols: usize, observations: &Box<[usize]>) -> HmmModel {
        let min_initial_value = 45.0;
        let max_initial_value = 55.0;
        let min_iterations = 200;
        let max_iterations = 200;
        let improvement_threshold = 0.001;

        let mut model = HmmModel {
            state_transition_matrix: vec![vec![0.0029371975966167385, 0.02271757809167469, 0.04654422249576483, 0.04077951517958343, 0.0012796918989409025, 0.012687231112356949, 0.02372914406893274, 0.004350952456399069, 0.03528293378508489, 0.00162094307199181, 0.011102850666049165, 0.10414254549000013, 0.03594106104739735, 0.19392816662807277, 0.002595946423565831, 0.020743196304737296, 0.0006215646366284385, 0.11400226688279241, 0.09948690448623417, 0.14511706133989835, 0.01250441798393682, 0.021450073734628462, 0.009969409269844366, 0.0026568841330392072, 0.032150735518153346, 0.0016575056976758358].into_boxed_slice(), vec![0.09309610324520737, 0.007016664578373638, 0.00269389800776845, 0.001440922190201729, 0.3028442551058765, 0.0010650294449317129, 0.0008770830722967047, 0.0016915173537150733, 0.061082571106377645, 0.006327527878711941, 0.0005638391179050244, 0.11984713695025687, 0.00269389800776845, 0.0007517854905400326, 0.11402079939857161, 0.0010023806540533768, 0.00031324395439168026, 0.05932840496178424, 0.018481393309109133, 0.008144342814183686, 0.10831975942864303, 0.0033203859165518105, 0.0022553564716200976, 0.00031324395439168026, 0.08213256484149856, 0.0003758927452700163].into_boxed_slice(), vec![0.13399844115354637, 0.0018706157443491816, 0.019236165237724083, 0.0025253312548713953, 0.15239282930631332, 0.0016212003117692907, 0.0010600155884645363, 0.16215120810600156, 0.06703039750584568, 0.00040530007794232267, 0.036102883865939205, 0.03647700701480904, 0.0017770849571317226, 0.0010911925175370225, 0.1988152766952455, 0.003055339049103663, 0.0010600155884645363, 0.0370381917381138, 0.009508963367108339, 0.09346843335931411, 0.028246297739672643, 0.0006547155105222135, 0.0017770849571317226, 0.0001558846453624318, 0.008261886204208885, 0.00021823850350740452].into_boxed_slice(), vec![0.10244019011709726, 0.03783755078197605, 0.022304240460954302, 0.02227768779373888, 0.16855633148349752, 0.02347255781843286, 0.01662196967685404, 0.027189931228591913, 0.12530203658957542, 0.004912243434853031, 0.0018586867050795252, 0.01991450041156634, 0.02862377525822469, 0.016675075011284884, 0.07841002628714054, 0.018586867050795253, 0.0016993707017869946, 0.03337670268978519, 0.06120389793154722, 0.10092668808581821, 0.03300496534876928, 0.006956798810440509, 0.030057619287857464, 0.00013276333607710895, 0.017285786357239586, 0.00037173734101590504].into_boxed_slice(), vec![0.08105897406825507, 0.017533487746805097, 0.04973355577511784, 0.0827677804953109, 0.037018740180437, 0.024579277280163268, 0.014715171933461831, 0.015565525842660232, 0.03165746124815757, 0.002607751988208426, 0.0048024749348062005, 0.04306840084873419, 0.038662757738220574, 0.1016780317141515, 0.027097944573122337, 0.02897682178201785, 0.0031827532029997246, 0.14204959587942792, 0.11036783880529324, 0.06223780754466383, 0.0074669171836278525, 0.019558139911563195, 0.02925217447642495, 0.011467629861189847, 0.01232608237904728, 0.0005669026061322664].into_boxed_slice(), vec![0.10180327146069397, 0.010140646355980778, 0.022618050350513647, 0.01053745425686698, 0.08258013315109564, 0.05916846699880958, 0.009920197522155108, 0.017724086239583794, 0.10832855694193377, 0.0036153608747409726, 0.00185177020413562, 0.026542039592610554, 0.015916405802213308, 0.008685684052731362, 0.18394250694413827, 0.014240994665138222, 0.0006613465014770072, 0.08002292667871787, 0.022353511749922842, 0.15995767382390547, 0.037079493849477535, 0.0023808474053172257, 0.013359199329835545, 0.00022044883382566906, 0.005775759446232529, 0.0005731669679467395].into_boxed_slice(), vec![0.10664407475867735, 0.012733620866707743, 0.013298418566440748, 0.008420620250564798, 0.16050523721503387, 0.014684740193058123, 0.01494146642020949, 0.11414048059149723, 0.0918052988293284, 0.0027726432532347504, 0.0012836311357568289, 0.030499075785582256, 0.015095502156500308, 0.03008831382214007, 0.098120764017252, 0.012939001848428836, 0.0006674881905935511, 0.09581022797288971, 0.03835489833641405, 0.07943109468063257, 0.03224481413021154, 0.0024132265352228384, 0.01591702608338468, 0.0002567262271513658, 0.0066235366605052375, 0.00030807147258163895].into_boxed_slice(), vec![0.16200396634383965, 0.0045439667289215785, 0.007759400812522865, 0.00294587673527543, 0.4656410651366078, 0.0036390242023990604, 0.002040934208752912, 0.005968769855786819, 0.12619134720911873, 0.0008279261412865587, 0.0006353851781966613, 0.003735294683944009, 0.0065849009376744904, 0.007451335271579028, 0.09315131794289234, 0.003927835647033907, 0.00021179505939888712, 0.01971619462040549, 0.010628261162562335, 0.04126152839016501, 0.015480293432427749, 0.001097483489612415, 0.00812522864239367, 0.00009627048154494869, 0.006161310818876716, 0.00017328686678090764].into_boxed_slice(), vec![0.031066686721452696, 0.01039202548780988, 0.07716078924698837, 0.03724720714315015, 0.04119891157206733, 0.018008286272954754, 0.028413985478511754, 0.001449414081194536, 0.0009981813955396333, 0.0003418429436779566, 0.005127644155169349, 0.05433935432704798, 0.030218916221131365, 0.24771307070679446, 0.07694200976303447, 0.01015957228610887, 0.0008887916535626871, 0.03619433087662204, 0.1279449769597856, 0.12213364691726034, 0.001449414081194536, 0.029986463019430355, 0.0015724775409186004, 0.002023710226573503, 0.00010938974197694612, 0.0069189011800418415].into_boxed_slice(), vec![0.1486810551558753, 0.0057553956834532375, 0.005275779376498801, 0.004316546762589928, 0.15587529976019185, 0.0028776978417266188, 0.0038369304556354917, 0.0057553956834532375, 0.028297362110311752, 0.0038369304556354917, 0.002398081534772182, 0.003357314148681055, 0.004316546762589928, 0.0028776978417266188, 0.294484412470024, 0.003357314148681055, 0.002398081534772182, 0.026378896882494004, 0.004796163069544364, 0.003357314148681055, 0.2729016786570743, 0.002398081534772182, 0.005275779376498801, 0.002398081534772182, 0.002398081534772182, 0.002398081534772182].into_boxed_slice(), vec![0.06977818853974121, 0.0118607516943931, 0.016019716574245224, 0.009242144177449169, 0.3285582255083179, 0.015403573629081947, 0.006469500924214418, 0.03126925446703635, 0.14525569932224275, 0.005083179297597043, 0.002618607516943931, 0.024953789279112754, 0.012322858903265557, 0.05730129390018484, 0.05560690080098583, 0.010166358595194085, 0.0012322858903265558, 0.010320394331484904, 0.08872458410351201, 0.04882932840418977, 0.006315465187923599, 0.0018484288354898336, 0.024029574861367836, 0.0007701786814540973, 0.015249537892791128, 0.0007701786814540973].into_boxed_slice(), vec![0.11746211852587286, 0.015034394723778455, 0.013568777628064204, 0.06306881308654233, 0.1658274826844432, 0.017634683119400515, 0.00520057679124412, 0.0076117533035482115, 0.1276032432687989, 0.001040115358248824, 0.005389688674562088, 0.13597144410561898, 0.014372503132165567, 0.005011464907926152, 0.07807956882490603, 0.014041557336359124, 0.000661891591612888, 0.00898281445760348, 0.04089544476751058, 0.03680590029075952, 0.02352079048767227, 0.0075171973618892276, 0.01016476372834078, 0.00011819492707373, 0.084131149091081, 0.00028366782497695197].into_boxed_slice(), vec![0.18915635203887918, 0.04073961576429493, 0.007517655099096363, 0.003037436403675298, 0.24098261067658897, 0.005695193256891184, 0.001822461842205179, 0.006606424177993773, 0.11276482648644544, 0.001328878426607943, 0.0007593591009188245, 0.003872731414686005, 0.04165084668539752, 0.004290378920191358, 0.12070012909104716, 0.06348242083681373, 0.0003037436403675298, 0.027678639228491154, 0.033183992710152634, 0.02729895967803174, 0.04267598147163794, 0.0007973270559647657, 0.008542789885336776, 0.00018983977522970614, 0.014655630647733313, 0.0002657756853215886].into_boxed_slice(), vec![0.07409360214233186, 0.011474795687585852, 0.05305878924949008, 0.1426787473463668, 0.0893008283498217, 0.01567898322487547, 0.11080739826004912, 0.013875206393695107, 0.06078727921077826, 0.0033577999472742155, 0.007506486658989052, 0.012737439469412108, 0.013167570867616656, 0.01868990301230731, 0.06746125348614561, 0.010142775873791122, 0.0011932677498577792, 0.007006979228816029, 0.06998654104979811, 0.1669742337417269, 0.012057554356121047, 0.006438095766674529, 0.01633111792537914, 0.0002497537150865119, 0.014222086553537483, 0.0007215107324721455].into_boxed_slice(), vec![0.01898014750588173, 0.018477887335113272, 0.02155753522429882, 0.02400274921356631, 0.008194771207274841, 0.11340505961035184, 0.012411113167146898, 0.009820508075814851, 0.013640328848238124, 0.0014935631393904148, 0.008617727140553543, 0.04451611197758334, 0.06455364931666183, 0.18047000978085595, 0.028998916175420972, 0.028959264056676096, 0.0005154775436834175, 0.13779111263845198, 0.04142324671548283, 0.06069417642549367, 0.09178143752147823, 0.022231621242961748, 0.04022046578022152, 0.0011895635623463481, 0.0055248618784530384, 0.0005286949165983769].into_boxed_slice(), vec![0.1317783971443939, 0.0033250207813798837, 0.002347073492738741, 0.0020536893061463986, 0.17304777272505012, 0.004498557527749254, 0.0012713314752334848, 0.032370055254021805, 0.06004596352256614, 0.0007334604664808567, 0.0009290499242090852, 0.09671898684660897, 0.009437191335387023, 0.000880152559777028, 0.13515231529020585, 0.05202679575570877, 0.00029338418659234265, 0.17783971443939173, 0.0230306586474989, 0.04317637279350643, 0.040046941469854776, 0.0003911789154564569, 0.004498557527749254, 0.0002444868221602856, 0.0036184049679722262, 0.0002444868221602856].into_boxed_slice(), vec![0.0044964028776978415, 0.0044964028776978415, 0.00539568345323741, 0.0044964028776978415, 0.00539568345323741, 0.0044964028776978415, 0.0044964028776978415, 0.00539568345323741, 0.0044964028776978415, 0.0044964028776978415, 0.0044964028776978415, 0.0044964028776978415, 0.0044964028776978415, 0.0044964028776978415, 0.0044964028776978415, 0.00539568345323741, 0.0044964028776978415, 0.0044964028776978415, 0.0044964028776978415, 0.00539568345323741, 0.8812949640287769, 0.00539568345323741, 0.0044964028776978415, 0.00539568345323741, 0.0044964028776978415, 0.0044964028776978415].into_boxed_slice(), vec![0.10308531089470864, 0.010258107213765718, 0.024865273707100313, 0.033390060193501624, 0.22561532885821436, 0.011896883174182976, 0.015694431313226814, 0.012101730169235133, 0.10138350508965996, 0.0018751378777851313, 0.015127162711543915, 0.01974409883079638, 0.02945069490403706, 0.02650404966751757, 0.10138350508965996, 0.013992625508178123, 0.0006460559074721881, 0.01878289370016703, 0.07889760801739623, 0.07989032807034131, 0.02018530774321641, 0.009706596073240679, 0.013094450222180202, 0.0001733320727364407, 0.031861586461189376, 0.00039393652894645614].into_boxed_slice(), vec![0.09845728964052919, 0.020638762268185125, 0.035932233589214393, 0.013185051002984454, 0.10870243062257791, 0.019287591500987394, 0.007023118383346449, 0.05388350235341282, 0.09723975114701035, 0.00267264547357793, 0.007438863234791905, 0.01717917118294258, 0.02348958410666825, 0.015219231168985435, 0.083921067870347, 0.038560334971566024, 0.00203418016600098, 0.010438165377362692, 0.07069147277613624, 0.19777576504476682, 0.03536800843368127, 0.002687493503986696, 0.03082451112859879, 0.0002524165169490267, 0.006874638079258786, 0.00022272045613149416].into_boxed_slice(), vec![0.06887022487130859, 0.010295312923327011, 0.011454890273638581, 0.0055377946356001085, 0.10546735302086155, 0.008615551341099974, 0.003500406393931184, 0.31280411812516934, 0.11806014630181523, 0.0015713898672446492, 0.0015605526957464101, 0.014966133839068004, 0.011238146843673802, 0.0056136548360877815, 0.11214305066377675, 0.007867786507721485, 0.0003793010024383636, 0.038081820644811706, 0.043034408019506906, 0.05342725548631807, 0.0214467623950149, 0.001332972094283392, 0.020872392305608237, 0.00006502302898943376, 0.021230018965050123, 0.0005635329179084259].into_boxed_slice(), vec![0.035906167234203555, 0.03185773741959894, 0.04903518728717367, 0.028717366628830874, 0.04309496783957624, 0.0061293984108967085, 0.03658721150208097, 0.0014755959137343927, 0.026598562239878925, 0.0009837306091562618, 0.0018917896329928112, 0.0968974650018918, 0.03658721150208097, 0.1388951948543322, 0.0035944003026863415, 0.04074914869466515, 0.00022701475595913735, 0.1453651153991676, 0.13984108967082862, 0.12750662126371548, 0.0004161937192584185, 0.00170261066969353, 0.0016269390843738176, 0.0012864169504351116, 0.002497162315550511, 0.0005297010972379871].into_boxed_slice(), vec![0.08454198473282443, 0.00200381679389313, 0.001049618320610687, 0.0018129770992366412, 0.6177480916030534, 0.0011450381679389313, 0.0007633587786259542, 0.0014312977099236641, 0.21125954198473282, 0.0008587786259541985, 0.0011450381679389313, 0.0009541984732824427, 0.002480916030534351, 0.0006679389312977099, 0.049904580152671754, 0.0012404580152671756, 0.0005725190839694657, 0.0016221374045801526, 0.005057251908396947, 0.0022900763358778627, 0.002099236641221374, 0.0009541984732824427, 0.0026717557251908397, 0.00047709923664122136, 0.004770992366412214, 0.00047709923664122136].into_boxed_slice(), vec![0.18162500698987866, 0.00682212156796958, 0.0051445506906000115, 0.005759660012302187, 0.1719510149303808, 0.004361684281160879, 0.0018453279651065259, 0.17804618911815692, 0.1863781244757591, 0.001677570877369569, 0.0018453279651065259, 0.006933959626460885, 0.009058882737795673, 0.035732259687971814, 0.12044958899513504, 0.006262931275513057, 0.0006151093217021753, 0.01627243751048482, 0.022535368785997874, 0.017390818095397866, 0.0017334899066152212, 0.0013979757311413073, 0.00631885030475871, 0.0002795951462282615, 0.009170720796286976, 0.00039143320471956607].into_boxed_slice(), vec![0.11284807034684904, 0.017586712261846604, 0.0986809965803615, 0.006350757205666829, 0.07278944797264289, 0.01172447484123107, 0.0039081582804103565, 0.023937469467513434, 0.09330727894479726, 0.0029311187103077674, 0.004885197850512946, 0.010258915486077186, 0.01074743527112848, 0.0039081582804103565, 0.032730825598436736, 0.2598925256472887, 0.003419638495359062, 0.008793356130923302, 0.01074743527112848, 0.1621885686370298, 0.008304836345872008, 0.005862237420615535, 0.01709819247679531, 0.003419638495359062, 0.011235955056179775, 0.002442598925256473].into_boxed_slice(), vec![0.10702360391479562, 0.04277489925158319, 0.04214162348877375, 0.028151986183074264, 0.08370754173862982, 0.033045480713874496, 0.014219919401266552, 0.038860103626943004, 0.06810592976396085, 0.006390328151986183, 0.0037996545768566492, 0.023949337938975246, 0.04179620034542314, 0.020782959124928037, 0.10391479562464019, 0.03172135866436385, 0.0016119746689694876, 0.02498560736902706, 0.09884858952216465, 0.1158894645941278, 0.008175014392630972, 0.004778353483016695, 0.05043177892918826, 0.0003454231433506045, 0.0034542314335060447, 0.001093839953943581].into_boxed_slice(), vec![0.17531305903398928, 0.009838998211091235, 0.011627906976744186, 0.008050089445438283, 0.40518783542039355, 0.008944543828264758, 0.008050089445438283, 0.017889087656529516, 0.08586762075134168, 0.005366726296958855, 0.011627906976744186, 0.025044722719141325, 0.012522361359570662, 0.004472271914132379, 0.03667262969588551, 0.007155635062611807, 0.004472271914132379, 0.008944543828264758, 0.014311270125223614, 0.016100178890876567, 0.01699463327370304, 0.006261180679785331, 0.009838998211091235, 0.004472271914132379, 0.01520572450805009, 0.06976744186046512].into_boxed_slice()].into_boxed_slice(),
            observation_probability_matrix: vec![
                vec![0.0;
                     number_of_observation_symbols
                ].into_boxed_slice();
                number_of_hidden_state_symbols
            ].into_boxed_slice(),
            initial_state_distribution_vector: vec![0.0; number_of_hidden_state_symbols].into_boxed_slice(),
            log_probability: std::f64::NEG_INFINITY,
            alpha_matrix: vec![
                vec![0.0;
                     number_of_hidden_state_symbols
                ].into_boxed_slice();
                observations.len()
            ].into_boxed_slice(),
            beta_matrix: vec![
                vec![0.0;
                     number_of_hidden_state_symbols
                ].into_boxed_slice();
                observations.len()
            ].into_boxed_slice(),
            gamma_matrix: vec![
                vec![0.0;
                     number_of_hidden_state_symbols
                ].into_boxed_slice();
                observations.len()
            ].into_boxed_slice(),
            di_gamma_tensor: vec![
                vec![
                    vec![0.0;
                         number_of_hidden_state_symbols
                    ].into_boxed_slice();
                    number_of_hidden_state_symbols
                ].into_boxed_slice();
                observations.len()
            ].into_boxed_slice(),
            scale_factors: vec![0.0; observations.len()].into_boxed_slice(),
        };

        // Generate guesses
        let mut rng = rand::thread_rng();
        for i in 0..number_of_hidden_state_symbols {
            for j in 0..number_of_observation_symbols {
                model.observation_probability_matrix[i][j] = rng.gen_range(min_initial_value, max_initial_value);
            }

            model.initial_state_distribution_vector[i] = rng.gen_range(min_initial_value, max_initial_value);
        }

        // Normalize
        for i in 0..number_of_hidden_state_symbols {
            let observation_probability_row_sum = model.observation_probability_matrix[i].iter().sum::<f64>();
            model.observation_probability_matrix[i] = model.observation_probability_matrix[i].iter().map(|&probability| probability / observation_probability_row_sum).collect();
        }
        let initial_state_distribution_vector_sum = model.initial_state_distribution_vector.iter().sum::<f64>();
        model.initial_state_distribution_vector = model.initial_state_distribution_vector.iter().map(|&probability| probability / initial_state_distribution_vector_sum).collect();

        let mut iterations = 0;
        let mut log_probability = std::f64::NEG_INFINITY;
        let mut old_log_probability = std::f64::NEG_INFINITY;

        while iterations < min_iterations || (log_probability - old_log_probability).abs() > improvement_threshold && iterations < max_iterations {
            old_log_probability = log_probability;

            model.populate_alpha_matrix_and_scale_factors(&observations);
            model.populate_beta_matrix(&observations);
            model.compute_gamma_matrix_and_di_gamma_tensor(&observations);

            model.initial_state_distribution_vector = model.gamma_matrix[0].clone();
            for i in 0..number_of_hidden_state_symbols {
                for j in 0..number_of_observation_symbols {
                    let mut numerator = 0.0;
                    let mut denominator = 0.0;
                    for observation_index in 0..(observations.len() - 1) {
                        if observations[observation_index] == j {
                            numerator += model.gamma_matrix[observation_index][i];
                        }
                        denominator += model.gamma_matrix[observation_index][i];
                    }

                    model.observation_probability_matrix[i][j] = numerator / denominator;
                }
            }

            log_probability = -(model.scale_factors.iter().map(|&scalar| scalar.log2())).sum::<f64>();
            iterations += 1;
        }

        model.log_probability = log_probability;

        return model;
    }

    pub fn get_number_of_hidden_state_symbols(&self) -> usize {
        return self.initial_state_distribution_vector.len();
    }

    pub fn get_number_of_observation_symbols(&self) -> usize {
        return match self.observation_probability_matrix.len() {
            0 => 0,
            _ => self.observation_probability_matrix[0].len(),
        };
    }

    fn populate_alpha_matrix_and_scale_factors(&mut self, observations: &Box<[usize]>) {
        self.scale_factors[0] = 0.0;
        for i in 0..self.get_number_of_hidden_state_symbols() {
            // alpha_0(i) = pi_i * b_i(O_0)
            self.alpha_matrix[0][i] = self.initial_state_distribution_vector[i] * self.observation_probability_matrix[i][observations[0]];
            self.scale_factors[0] += self.alpha_matrix[0][i]
        }

        self.scale_factors[0] = 1.0 / self.scale_factors[0];
        self.alpha_matrix[0] = self.alpha_matrix[0].iter().map(|&alpha_value| self.scale_factors[0] * alpha_value).collect();

        for observation_index in 1..observations.len() {
            self.scale_factors[observation_index] = 0.0;
            for i in 0..self.get_number_of_hidden_state_symbols() {
                // += alpha_t-1(j) * a_ji
                self.alpha_matrix[observation_index][i] = self.alpha_matrix[observation_index - 1]
                    .iter()
                    .enumerate()
                    .map(|alpha_value_pair| alpha_value_pair.1 * self.state_transition_matrix[alpha_value_pair.0][i])
                    .sum::<f64>();

                // = sum(alpha_t-1(j) * a_ji) * b_i(O_t)
                self.alpha_matrix[observation_index][i] *= self.observation_probability_matrix[i][observations[observation_index]];
                self.scale_factors[observation_index] += self.alpha_matrix[observation_index][i];
            }

            self.scale_factors[observation_index] = 1.0 / self.scale_factors[observation_index];
            for i in 0..self.get_number_of_hidden_state_symbols() {
                self.alpha_matrix[observation_index][i] *= self.scale_factors[observation_index];
            }
        }
    }

    fn populate_beta_matrix(&mut self, observations: &Box<[usize]>) {
        // beta_T-1(i) = c_T-1
        self.beta_matrix[observations.len() - 1] = vec![
            self.scale_factors[observations.len() - 1];
            self.get_number_of_hidden_state_symbols()
        ].into_boxed_slice();

        // From T-2 to 0
        for observation_index in (0..(observations.len() - 1)).rev() {
            for i in 0..self.get_number_of_hidden_state_symbols() {
                self.beta_matrix[observation_index][i] = 0.0;
                for j in 0..self.get_number_of_hidden_state_symbols() {
                    // += a_ij * b_j(O_t+1) * beta_t+1(j)
                    self.beta_matrix[observation_index][i] += self.state_transition_matrix[i][j] * self.observation_probability_matrix[j][observations[observation_index + 1]] * self.beta_matrix[observation_index + 1][j];
                }

                self.beta_matrix[observation_index][i] *= self.scale_factors[observation_index];
            }
        }
    }

    fn compute_gamma_matrix_and_di_gamma_tensor(&mut self, observations: &Box<[usize]>) {
        // From 0 to T-2
        for observation_index in 0..(observations.len() - 1) {
            let mut denominator = 0.0;
            for i in 0..self.get_number_of_hidden_state_symbols() {
                for j in 0..self.get_number_of_hidden_state_symbols() {
                    // += alpha_t(t) * a_ij * b_j(O_t+1) * beta_t+1(j)
                    denominator += self.alpha_matrix[observation_index][i] * self.state_transition_matrix[i][j] * self.observation_probability_matrix[j][observations[observation_index + 1]] * self.beta_matrix[observation_index + 1][j];
                }
            }

            for i in 0..self.get_number_of_hidden_state_symbols() {
                self.gamma_matrix[observation_index][i] = 0.0;
                for j in 0..self.get_number_of_hidden_state_symbols() {
                    // += (alpha_t(i) * a_ij * b_j(O_t+1) * beta_t+1(j)) / denom
                    self.di_gamma_tensor[observation_index][i][j] = (self.alpha_matrix[observation_index][i] * self.state_transition_matrix[i][j] * self.observation_probability_matrix[j][observations[observation_index + 1]] * self.beta_matrix[observation_index + 1][j]) / denominator;
                    // += di-gamma_t(i, j)
                    self.gamma_matrix[observation_index][i] += self.di_gamma_tensor[observation_index][i][j];
                }
            }
        }

        let denominator = self.alpha_matrix[observations.len() - 1].iter().sum::<f64>();
        self.gamma_matrix[observations.len() - 1] = self.alpha_matrix[observations.len() - 1].iter().map(|&alpha_value| alpha_value / denominator).collect();
    }
}

fn main() {
    let number_of_observation_symbols = 26;

    let args: Box<[String]> = env::args().collect();
    if args.len() != 3 {
        print_usage_and_panic();
    }
    let number_of_hidden_state_symbols = args[1].parse::<usize>().unwrap();
    let filename = &args[2];

    let path = Path::new(filename);

    let mut file = match File::open(&path) {
        Err(reason) => panic!("Couldn't open file: {:?}", reason),
        Ok(file) => file,
    };

    let mut raw_input = String::new();
    file.read_to_string(&mut raw_input).unwrap();

    let lowercase_input = raw_input.to_lowercase();
    let no_extra_spaces_no_new_lines = Regex::new("(\n\\s*)|(\\s+\\s+)").unwrap().replace_all(lowercase_input.as_str(), " ").to_string();
    let sanitized_input = Regex::new("[^a-z ]").unwrap().replace_all(no_extra_spaces_no_new_lines.as_str(), "");

    // a, b, c, ..., z, SPACE => 0, 1, 2, ..., 25, 26
    let observations: Box<[usize]> = sanitized_input.chars().map(|ch| match ch {
        ' ' => 26 as usize,
        _ => ch as usize - 'a' as usize,
    }).collect();

    let mut best_models = Vec::new();
    for t in [1000, 400, 300].iter() {
        for n in [1, 10, 100, 1000].iter() {
            let mut best_model = HmmModel {
                state_transition_matrix: Vec::new().into_boxed_slice(),
                observation_probability_matrix: Vec::new().into_boxed_slice(),
                initial_state_distribution_vector: Vec::new().into_boxed_slice(),
                log_probability: std::f64::NEG_INFINITY,
                alpha_matrix: Vec::new().into_boxed_slice(),
                beta_matrix: Vec::new().into_boxed_slice(),
                gamma_matrix: Vec::new().into_boxed_slice(),
                di_gamma_tensor: Vec::new().into_boxed_slice(),
                scale_factors: Vec::new().into_boxed_slice(),
            };

            for i in 0..*n {
                if i % (1 << 7) == 0 {
                    println!("i/n: {:?}/{:?}, ", i, n);
                }
                let observations_slice = &observations[0..*t];


                let time_before_training = SystemTime::now();
                let model = HmmModel::train_model(number_of_hidden_state_symbols, number_of_observation_symbols, &observations_slice.to_vec().into_boxed_slice());

                if model.log_probability > best_model.log_probability {
                    best_model = model;
                }
            }

            best_models.push(best_model);
        }
    }

    for model in best_models.iter() {
        let mut presumed_key_list = Vec::with_capacity(26);
        for j in 0..number_of_observation_symbols {
            match j {
                26 => print!("SPACE    "),
                _ => print!("{:?}      ", (j as u8 + 'a' as u8) as char),
            }

            let mut state_max_probability = (0, 0.0);
            for i in 0..number_of_hidden_state_symbols {
                print!("{:.*}   ", 5, model.observation_probability_matrix[i][j]);

                if model.observation_probability_matrix[i][j] > state_max_probability.1 {
                    state_max_probability = (i, model.observation_probability_matrix[i][j]);
                }
            }

            presumed_key_list.push(state_max_probability.0);
            println!("{:?}", state_max_probability.0);
        }

        let presumed_key = String::from_utf8(presumed_key_list
            .iter()
            .map(|val| *val as u8 + 'a' as u8)
            .collect::<Vec<u8>>())
            .unwrap();
        let actual_key = "cweljndfoqrvaumstxhygipbkz";
        let score = presumed_key
            .chars()
            .enumerate()
            .fold(0, |sum, pair| if pair.1 as u8 == actual_key.as_bytes()[pair.0] {
                sum + 1
            } else {
                sum
            });

        println!("Actual key:   {:?}", actual_key);
        println!("Presumed key: {:?}", presumed_key);
        println!("Score: {:?}/26 = {:.*}", score, 4, score as f64 / 26.0);
        println!("Log probability: {:.*}", 5, model.log_probability);
    }
}

fn print_usage_and_panic() {
    println!("Usage: cargo run <number_of_hidden_state_symbols> <input_file>");
    panic!("Incorrect command arguments");
}