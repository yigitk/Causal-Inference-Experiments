from math import *
import numpy as np
from collections import namedtuple
M_PI   =    3.14159265358979323846264338328
M_LNPI   =  1.14472988584940017414342735135
GSL_DBL_EPSILON   =     2.2204460492503131e-16
M_E   =     2.71828182845904523536028747135
LogRootTwoPi_  = 0.9189385332046727418
M_EULER  =  0.57721566490153286060651209008    
GSL_LOG_DBL_MIN  = (-7.0839641853226408e+02)
GSL_LOG_DBL_MAX  =  7.0978271289338397e+02
GSL_SQRT_DBL_MAX =  1.3407807929942596e+154
GSL_SQRT_DBL_MIN =  1.4916681462400413e-154
GSL_SF_FACT_NMAX= 170

def GSL_IS_ODD(n):
    return n % 1 == 1

def GSL_IS_EVEN(n):
    return n % 1 == 0

def GSL_SIGN(x):
    if x >= 0.0:
        return 1
    return 0

def GSL_ERROR_SELECT_2(a,b):
    return a if a != GSL_SUCCESS else (b if b != GSL_SUCCESS else GSL_SUCCESS)

def GSL_ERROR_SELECT_3(a,b,c):
    return a if a != GSL_SUCCESS else GSL_ERROR_SELECT_2(b,c)

GSL_SUCCESS = 0
GSL_FAILURE  = -1
GSL_CONTINUE = -2
GSL_EDOM     = 1
GSL_ERANGE   = 2
GSL_EFAULT   = 3
GSL_EINVAL   = 4
GSL_EFAILED  = 5
GSL_EFACTOR  = 6
GSL_ESANITY  = 7
GSL_ENOMEM   = 8
GSL_EBADFUNC = 9
GSL_ERUNAWAY = 1
GSL_EMAXITER = 11
GSL_EZERODIV = 12
GSL_EBADTOL  = 13
GSL_ETOL     = 14
GSL_EUNDRFLW = 15
GSL_EOVRFLW  = 16
GSL_ELOSS    = 17
GSL_EROUND   = 18
GSL_EBADLEN  = 19
GSL_ENOTSQR  = 20
GSL_ESING    = 21
GSL_EDIVERGE = 22
GSL_EUNSUP   = 23
GSL_EUNIMPL  = 24
GSL_ECACHE   = 25
GSL_ETABLE   = 26
GSL_ENOPROG  = 27
GSL_ENOPROGJ = 28
GSL_ETOLF    = 29
GSL_ETOLX    = 30
GSL_ETOLG    = 31
GSL_EOF      = 32

class cheb_series:
    def __init__(self, c, order, a, b, order_sp):
        self.c = c
        self.order = order
        self.a = a
        self.b = b
        self.order_sp = order_sp

FactStruct = namedtuple("FactStruct", "n f i")
fact_table = [
FactStruct( 0,  1.0,     1 ),
FactStruct( 1,  1.0,     1 ),
FactStruct( 2,  2.0,     2 ),
FactStruct( 3,  6.0,     6 ),
FactStruct( 4,  24.0,    24 ),
FactStruct( 5,  120.0,   120 ),
FactStruct( 6,  720.0,   720 ),
FactStruct( 7,  5040.0,  5040 ),
FactStruct( 8,  40320.0, 40320 ),

FactStruct( 9,  362880.0,     362880 ),
FactStruct( 10, 3628800.0,    3628800 ),
FactStruct( 11, 39916800.0,   39916800 ),
FactStruct( 12, 479001600.0,  479001600 ),

FactStruct( 13, 6227020800.0,                               0 ),
FactStruct( 14, 87178291200.0,                              0 ),
FactStruct( 15, 1307674368000.0,                            0 ),
FactStruct( 16, 20922789888000.0,                           0 ),
FactStruct( 17, 355687428096000.0,                          0 ),
FactStruct( 18, 6402373705728000.0,                         0 ),
FactStruct( 19, 121645100408832000.0,                       0 ),
FactStruct( 20, 2432902008176640000.0,                      0 ),
FactStruct( 21, 51090942171709440000.0,                     0 ),
FactStruct( 22, 1124000727777607680000.0,                   0 ),
FactStruct( 23, 25852016738884976640000.0,                  0 ),
FactStruct( 24, 620448401733239439360000.0,                 0 ),
FactStruct( 25, 15511210043330985984000000.0,               0 ),
FactStruct( 26, 403291461126605635584000000.0,              0 ),
FactStruct( 27, 10888869450418352160768000000.0,            0 ),
FactStruct( 28, 304888344611713860501504000000.0,           0 ),
FactStruct( 29, 8841761993739701954543616000000.0,          0 ),
FactStruct( 30, 265252859812191058636308480000000.0,        0 ),
FactStruct( 31, 8222838654177922817725562880000000.0,       0 ),
FactStruct( 32, 263130836933693530167218012160000000.0,     0 ),
FactStruct( 33, 8683317618811886495518194401280000000.0,    0 ),
FactStruct( 34, 2.95232799039604140847618609644e38,  0 ),
FactStruct( 35, 1.03331479663861449296666513375e40,  0 ),
FactStruct( 36, 3.71993326789901217467999448151e41,  0 ),
FactStruct( 37, 1.37637530912263450463159795816e43,  0 ),
FactStruct( 38, 5.23022617466601111760007224100e44,  0 ),
FactStruct( 39, 2.03978820811974433586402817399e46,  0 ),
FactStruct( 40, 8.15915283247897734345611269600e47,  0 ),
FactStruct( 41, 3.34525266131638071081700620534e49,  0 ),
FactStruct( 42, 1.40500611775287989854314260624e51,  0 ),
FactStruct( 43, 6.04152630633738356373551320685e52,  0 ),
FactStruct( 44, 2.65827157478844876804362581101e54,  0 ),
FactStruct( 45, 1.19622220865480194561963161496e56,  0 ),
FactStruct( 46, 5.50262215981208894985030542880e57,  0 ),
FactStruct( 47, 2.58623241511168180642964355154e59,  0 ),
FactStruct( 48, 1.24139155925360726708622890474e61,  0 ),
FactStruct( 49, 6.08281864034267560872252163321e62,  0 ),
FactStruct( 50, 3.04140932017133780436126081661e64,  0 ),
FactStruct( 51, 1.55111875328738228022424301647e66,  0 ),
FactStruct( 52, 8.06581751709438785716606368564e67,  0 ),
FactStruct( 53, 4.27488328406002556429801375339e69,  0 ),
FactStruct( 54, 2.30843697339241380472092742683e71,  0 ),
FactStruct( 55, 1.26964033536582759259651008476e73,  0 ),
FactStruct( 56, 7.10998587804863451854045647464e74,  0 ),
FactStruct( 57, 4.05269195048772167556806019054e76,  0 ),
FactStruct( 58, 2.35056133128287857182947491052e78,  0 ),
FactStruct( 59, 1.38683118545689835737939019720e80,  0 ),
FactStruct( 60, 8.32098711274139014427634118320e81,  0 ),
FactStruct( 61, 5.07580213877224798800856812177e83,  0 ),
FactStruct( 62, 3.14699732603879375256531223550e85,  0 ),
FactStruct( 63, 1.982608315404440064116146708360e87,  0 ),
FactStruct( 64, 1.268869321858841641034333893350e89,  0 ),
FactStruct( 65, 8.247650592082470666723170306800e90,  0 ),
FactStruct( 66, 5.443449390774430640037292402480e92,  0 ),
FactStruct( 67, 3.647111091818868528824985909660e94,  0 ),
FactStruct( 68, 2.480035542436830599600990418570e96,  0 ),
FactStruct( 69, 1.711224524281413113724683388810e98,  0 ),
FactStruct( 70, 1.197857166996989179607278372170e100,  0 ),
FactStruct( 71, 8.504785885678623175211676442400e101,  0 ),
FactStruct( 72, 6.123445837688608686152407038530e103,  0 ),
FactStruct( 73, 4.470115461512684340891257138130e105,  0 ),
FactStruct( 74, 3.307885441519386412259530282210e107,  0 ),
FactStruct( 75, 2.480914081139539809194647711660e109,  0 ),
FactStruct( 76, 1.885494701666050254987932260860e111,  0 ),
FactStruct( 77, 1.451830920282858696340707840860e113,  0 ),
FactStruct( 78, 1.132428117820629783145752115870e115,  0 ),
FactStruct( 79, 8.946182130782975286851441715400e116,  0 ),
FactStruct( 80, 7.156945704626380229481153372320e118,  0 ),
FactStruct( 81, 5.797126020747367985879734231580e120,  0 ),
FactStruct( 82, 4.753643337012841748421382069890e122,  0 ),
FactStruct( 83, 3.945523969720658651189747118010e124,  0 ),
FactStruct( 84, 3.314240134565353266999387579130e126,  0 ),
FactStruct( 85, 2.817104114380550276949479442260e128,  0 ),
FactStruct( 86, 2.422709538367273238176552320340e130,  0 ),
FactStruct( 87, 2.107757298379527717213600518700e132,  0 ),
FactStruct( 88, 1.854826422573984391147968456460e134,  0 ),
FactStruct( 89, 1.650795516090846108121691926250e136,  0 ),
FactStruct( 90, 1.485715964481761497309522733620e138,  0 ),
FactStruct( 91, 1.352001527678402962551665687590e140,  0 ),
FactStruct( 92, 1.243841405464130725547532432590e142,  0 ),
FactStruct( 93, 1.156772507081641574759205162310e144,  0 ),
FactStruct( 94, 1.087366156656743080273652852570e146,  0 ),
FactStruct( 95, 1.032997848823905926259970209940e148,  0 ),
FactStruct( 96, 9.916779348709496892095714015400e149,  0 ),
FactStruct( 97, 9.619275968248211985332842594960e151,  0 ),
FactStruct( 98, 9.426890448883247745626185743100e153,  0 ),
FactStruct( 99, 9.332621544394415268169923885600e155,  0 ),
FactStruct( 100, 9.33262154439441526816992388563e157,  0 ),
FactStruct( 101, 9.42594775983835942085162312450e159,  0 ),
FactStruct( 102, 9.61446671503512660926865558700e161,  0 ),
FactStruct( 103, 9.90290071648618040754671525458e163,  0 ),
FactStruct( 104, 1.02990167451456276238485838648e166,  0 ),
FactStruct( 105, 1.08139675824029090050410130580e168,  0 ),
FactStruct( 106, 1.146280563734708354534347384148e170,  0 ),
FactStruct( 107, 1.226520203196137939351751701040e172,  0 ),
FactStruct( 108, 1.324641819451828974499891837120e174,  0 ),
FactStruct( 109, 1.443859583202493582204882102460e176,  0 ),
FactStruct( 110, 1.588245541522742940425370312710e178,  0 ),
FactStruct( 111, 1.762952551090244663872161047110e180,  0 ),
FactStruct( 112, 1.974506857221074023536820372760e182,  0 ),
FactStruct( 113, 2.231192748659813646596607021220e184,  0 ),
FactStruct( 114, 2.543559733472187557120132004190e186,  0 ),
FactStruct( 115, 2.925093693493015690688151804820e188,  0 ),
FactStruct( 116, 3.393108684451898201198256093590e190,  0 ),
FactStruct( 117, 3.96993716080872089540195962950e192,  0 ),
FactStruct( 118, 4.68452584975429065657431236281e194,  0 ),
FactStruct( 119, 5.57458576120760588132343171174e196,  0 ),
FactStruct( 120, 6.68950291344912705758811805409e198,  0 ),
FactStruct( 121, 8.09429852527344373968162284545e200,  0 ),
FactStruct( 122, 9.87504420083360136241157987140e202,  0 ),
FactStruct( 123, 1.21463043670253296757662432419e205,  0 ),
FactStruct( 124, 1.50614174151114087979501416199e207,  0 ),
FactStruct( 125, 1.88267717688892609974376770249e209,  0 ),
FactStruct( 126, 2.37217324288004688567714730514e211,  0 ),
FactStruct( 127, 3.01266001845765954480997707753e213,  0 ),
FactStruct( 128, 3.85620482362580421735677065923e215,  0 ),
FactStruct( 129, 4.97450422247728744039023415041e217,  0 ),
FactStruct( 130, 6.46685548922047367250730439554e219,  0 ),
FactStruct( 131, 8.47158069087882051098456875820e221,  0 ),
FactStruct( 132, 1.11824865119600430744996307608e224,  0 ),
FactStruct( 133, 1.48727070609068572890845089118e226,  0 ),
FactStruct( 134, 1.99294274616151887673732419418e228,  0 ),
FactStruct( 135, 2.69047270731805048359538766215e230,  0 ),
FactStruct( 136, 3.65904288195254865768972722052e232,  0 ),
FactStruct( 137, 5.01288874827499166103492629211e234,  0 ),
FactStruct( 138, 6.91778647261948849222819828311e236,  0 ),
FactStruct( 139, 9.61572319694108900419719561353e238,  0 ),
FactStruct( 140, 1.34620124757175246058760738589e241,  0 ),
FactStruct( 141, 1.89814375907617096942852641411e243,  0 ),
FactStruct( 142, 2.69536413788816277658850750804e245,  0 ),
FactStruct( 143, 3.85437071718007277052156573649e247,  0 ),
FactStruct( 144, 5.55029383273930478955105466055e249,  0 ),
FactStruct( 145, 8.04792605747199194484902925780e251,  0 ),
FactStruct( 146, 1.17499720439091082394795827164e254,  0 ),
FactStruct( 147, 1.72724589045463891120349865931e256,  0 ),
FactStruct( 148, 2.55632391787286558858117801578e258,  0 ),
FactStruct( 149, 3.80892263763056972698595524351e260,  0 ),
FactStruct( 150, 5.71338395644585459047893286526e262,  0 ),
FactStruct( 151, 8.62720977423324043162318862650e264,  0 ),
FactStruct( 152, 1.31133588568345254560672467123e267,  0 ),
FactStruct( 153, 2.00634390509568239477828874699e269,  0 ),
FactStruct( 154, 3.08976961384735088795856467036e271,  0 ),
FactStruct( 155, 4.78914290146339387633577523906e273,  0 ),
FactStruct( 156, 7.47106292628289444708380937294e275,  0 ),
FactStruct( 157, 1.17295687942641442819215807155e278,  0 ),
FactStruct( 158, 1.85327186949373479654360975305e280,  0 ),
FactStruct( 159, 2.94670227249503832650433950735e282,  0 ),
FactStruct( 160, 4.71472363599206132240694321176e284,  0 ),
FactStruct( 161, 7.59070505394721872907517857094e286,  0 ),
FactStruct( 162, 1.22969421873944943411017892849e289,  0 ),
FactStruct( 163, 2.00440157654530257759959165344e291,  0 ),
FactStruct( 164, 3.28721858553429622726333031164e293,  0 ),
FactStruct( 165, 5.42391066613158877498449501421e295,  0 ),
FactStruct( 166, 9.00369170577843736647426172359e297,  0 ),
FactStruct( 167, 1.50361651486499904020120170784e300,  0 ),
FactStruct( 168, 2.52607574497319838753801886917e302,  0 ),
FactStruct( 169, 4.26906800900470527493925188890e304,  0 ),
FactStruct( 170, 7.25741561530799896739672821113e306,  0 )
]
class gsl_sf_result:
    def __init__(self, val, err):
        self.val = val
        self.err = err

def GSL_ERROR_VAL(reason, gsl_errno, value):
    return

def EVAL_RESULT(fn, result):
    status = fn
    if status != GSL_SUCCESS:
        GSL_ERROR_VAL(fn, status, result.val)
    result_val = result.val
    return result_val

def cheb_eval_e(cs, x, result):
    d = 0.0
    dd = 0.0
    #added
    cs_a_IV = cs.a
    cs_b_IV = cs.b
    y = (2.0*x - cs_a_IV - cs_b_IV) / (cs_b_IV - cs_a_IV)
    y2 = 2.0 * y
    e = 0.0
    for j in range(cs.order, 0, -1):
        temp = d
        #add 
        cs_c_j_IV = cs.c[j]
        d = y2 * d - dd + cs_c_j_IV
        e += fabs(y2*temp) + fabs(dd) + fabs(cs_c_j_IV)
        dd = temp
    temp = d
    #add
    cs_c_0_IV = cs.c[0]
    d = y * d - dd + 0.5 * cs_c_0_IV
    e += fabs(y*temp) + fabs(dd) + 0.5 * fabs(cs_c_0_IV)
    result_val = d
    result.val = result_val
    #add
    cs_c_cs_order_IV = cs.c[cs.order]
    result_err = GSL_DBL_EPSILON * e + fabs(cs_c_cs_order_IV)
    result.err = result_err
    return GSL_SUCCESS

psics_data = [-.038057080835217922,
	.491415393029387130,
	-.056815747821244730,
	.008357821225914313,
	-.001333232857994342,
	.000220313287069308,
	-.000037040238178456,
	.000006283793654854,
	-.000001071263908506,
	.000000183128394654,
	-.000000031353509361,
	.000000005372808776,
	-.000000000921168141,
	.000000000157981265,
	-.000000000027098646,
	.000000000004648722,
	-.000000000000797527,
	.000000000000136827,
	-.000000000000023475,
	.000000000000004027,
	-.000000000000000691,
	.000000000000000118,
	-.000000000000000020]

psi_cs = cheb_series(psics_data,
	22,
	-1, 1,
	17)

apsics_data = [-.0204749044678185,
	-.0101801271534859,
	.0000559718725387,
	-.0000012917176570,
	.0000000572858606,
	-.0000000038213539,
	.0000000003397434,
	-.0000000000374838,
	.0000000000048990,
	-.0000000000007344,
	.0000000000001233,
	-.0000000000000228,
	.0000000000000045,
	-.0000000000000009,
	.0000000000000002,
	-.0000000000000000]

apsi_cs = cheb_series(apsics_data,
	15,
	-1, 1,
	9
)

def psi_x(x, result):
    y = fabs(x)
    if x == 0.0 or x == -1.0 or x == -2.0:
        print("domain error")
    elif y >= 2.0:
        t = 8.0 / (y*y) - 1.0
        result_c = gsl_sf_result(0, 0)
        cheb_eval_e(apsi_cs, t, result_c)
        if x < 0.0:
            s = sin(M_PI*x)
            c = cos(M_PI*x)
            if fabs(s) < 2.0*GSL_SQRT_DBL_MIN:
                print("domain error")
            else:
                result_c_val_IV = result_c.val
                result_val = log(y) - 0.5 / x + result_c_val_IV - M_PI * c / s
                result.val = result_val

                result_err = M_PI * fabs(x)*GSL_DBL_EPSILON / (s*s)
                result.err = result_err



                result_err = result.err
                result_c_err_IV = result_c.err
                result_err += result_c_err_IV
                result.err = result_err
                
                result_val_IV = result.val
                result_err = result.err
                result_err += GSL_DBL_EPSILON * fabs(result_val_IV)
                result.err = result_err
                return GSL_SUCCESS
        else:
            result_c_val_IV = result_c.val
            result_val = log(y) - 0.5 / x + result_c_val_IV
            result.val = result_val

            result_c_err_IV = result_c.err
            result_err = result_c_err_IV
            result.err = result_err

            result_err = result.err
            result_val_IV = result.val
            result_err += GSL_DBL_EPSILON * fabs(result_val_IV)
            result.err = result_err
            return GSL_SUCCESS
    else:
        result_c = gsl_sf_result(0, 0)
        if x < -1.0:
            v = x + 2.0
            t1 = 1.0 / x
            t2 = 1.0 / (x + 1.0)
            t3 = 1.0 / v
            cheb_eval_e(psi_cs, 2.0*v - 1.0, result_c)

            result_c_val_IV = result_c.val
            result_val = -(t1 + t2 + t3) + result_c_val_IV
            result.val = result_val

            result_err = GSL_DBL_EPSILON * (fabs(t1) + fabs(x / (t2*t2)) + fabs(x / (t3*t3)))
            result.err = result_err

            result_err = result.err
            result_c_err_IV = result_c.err
            result_err += result_c_err_IV
            result.err = result_err

            result_err = result.err
            result_val_IV = result.val
            result_err += GSL_DBL_EPSILON * fabs(result_val_IV)
            result.err = result_err
            return GSL_SUCCESS
        elif x < 0.0:
            v = x + 1.0
            t1 = 1.0 / x
            t2 = 1.0 / v
            cheb_eval_e(psi_cs, 2.0*v - 1.0, result_c)

            result_c_val_IV = result_c.val
            result_val = -(t1 + t2) + result_c_val_IV
            result.val = result_val


            result_err = GSL_DBL_EPSILON * (fabs(t1) + fabs(x / (t2*t2)))
            result.err = result_err


            result_err = result.err
            result_c_err_IV = result_c.err
            result_err += result_c_err_IV
            result.err = result_err

            result_err = result.err
            result_val_IV = result.val
            result_err += GSL_DBL_EPSILON * fabs(result_val_IV)
            result.err = result_err
            return GSL_SUCCESS
        elif x < 1.0:
            t1 = 1.0 / x
            cheb_eval_e(psi_cs, 2.0*x - 1.0, result_c)

            result_c_val_IV = result_c.val
            result_val = -t1 + result_c_val_IV
            result.val = result_val


            result_err = GSL_DBL_EPSILON * t1
            result.err = result_err

            result_err = result.err
            result_c_err_IV = result_c.err
            result_err += result_c_err_IV
            result.err = result_err

            result_err = result.err
            result_val_IV = result.val
            result_err += GSL_DBL_EPSILON * fabs(result_val_IV)
            result.err = result_err
            return GSL_SUCCESS
        else:
            v = x - 1.0
            return cheb_eval_e(psi_cs, 2.0*v - 1.0, result) 

hzeta_c = [1.00000000000000000000000000000,
	0.083333333333333333333333333333,
	-0.00138888888888888888888888888889,
	0.000033068783068783068783068783069,
	-8.2671957671957671957671957672e-07,
	2.0876756987868098979210090321e-08,
	-5.2841901386874931848476822022e-10,
	1.3382536530684678832826980975e-11,
	-3.3896802963225828668301953912e-13,
	8.5860620562778445641359054504e-15,
	-2.1748686985580618730415164239e-16,
	5.5090028283602295152026526089e-18,
	-1.3954464685812523340707686264e-19,
	3.5347070396294674716932299778e-21,
	-8.9535174270375468504026113181e-23]

def gsl_sf_hzeta_e(s, q, result):
    if s <= 1.0 or q <= 0.0:
        print("domain error")
    else:
        max_bits = 54.0
        ln_term0 = -s * log(q)
        if ln_term0 < GSL_LOG_DBL_MIN + 1.0:
            print("underflow error")
        elif ln_term0 > GSL_LOG_DBL_MAX - 1.0:
            print("overflow error")
        
        elif (s > max_bits and q < 1.0) or  (s > 0.5*max_bits and q < 0.25):

            result_val = pow(q, -s)
            result.val = result_val

            result_val_IV = result.val
            result_err = 2.0 * GSL_DBL_EPSILON * fabs(result_val_IV)
            result.err = result_err
            return GSL_SUCCESS

        elif s > 0.5*max_bits and q < 1.0:
            p1 = pow(q, -s)
            p2 = pow(q / (1.0 + q), s)
            p3 = pow(q / (2.0 + q), s)
            result_val = p1 * (1.0 + p2 + p3)
            result.val = result_val

            result_val_IV = result.val
            result_err = GSL_DBL_EPSILON * (0.5*s + 2.0) * fabs(result_val_IV)
            result.err = result_err
            return GSL_SUCCESS
        else:
            jmax = 12
            kmax = 10
            pmax = pow(kmax + q, -s)
            scp = s
            pcp = pmax / (kmax + q)
            ans = pmax * ((kmax + q) / (s - 1.0) + 0.5)
            for k in range(0, kmax):
                ans += pow(k + q, -s)
        
            for j in range(0, jmax+1):
                hzeta_c_j_1_IV = hzeta_c[j + 1]
                delta = hzeta_c_j_1_IV * scp * pcp
                ans += delta
                if fabs(delta / ans) < 0.5*GSL_DBL_EPSILON:
                    break
                scp *= (s + 2 * j + 1)*(s + 2 * j + 2)
                pcp /= (kmax + q)*(kmax + q)
        
            result_val = ans
            result.val = result_val

            result_err = 2.0 * (jmax + 1.0) * GSL_DBL_EPSILON * fabs(ans)
            result.err = result_err
            return GSL_SUCCESS

def gsl_sf_exp_mult_err_e(x, dx, y, dy, result):
    ay = fabs(y)
    if y == 0.0:
        result_val = 0.0
        result.val = result_val
        result_err = fabs(dy * exp(x))
        result.err = result_err
        return GSL_SUCCESS
    elif (x < 0.5*GSL_LOG_DBL_MAX   and   x > 0.5*GSL_LOG_DBL_MIN) and (ay < 0.8*GSL_SQRT_DBL_MAX  and  ay > 1.2*GSL_SQRT_DBL_MIN):

        ex = exp(x)
        result_val = y * ex
        result.val = result_val
        result_err = ex * (fabs(dy) + fabs(y*dx))
        result.err = result_err


        result_err = result.err
        result_val_IV = result.val
        result_err += 2.0 * GSL_DBL_EPSILON * fabs(result_val_IV)
        result.err = result_err
        return GSL_SUCCESS
    else:
        ly = log(ay)
        lnr = x + ly
        if lnr > GSL_LOG_DBL_MAX - 0.01:
            print("overflow error")
        elif lnr < GSL_LOG_DBL_MIN + 0.01:
            print("underflow error")
        else:
            sy = GSL_SIGN(y)
            M = floor(x)
            N = floor(ly)
            a = x - M
            b = ly - N
            eMN = exp(M + N)
            eab = exp(a + b)
            result_val = sy * eMN * eab
            result.val = result_val

            result_err = eMN * eab * 2.0*GSL_DBL_EPSILON
            result.err = result_err

            result_err = result.err
            result_err += eMN * eab * fabs(dy / y)
            result.err = result_err

            result_err = result.err
            result_err += eMN * eab * fabs(dx)
            result.err = result_err
            return GSL_SUCCESS

def gsl_sf_psi_e(x, result):
    return psi_x(x, result)

def psi_n_xg0(n, x, result):
    if n == 0:
        return gsl_sf_psi_e(x, result)
    else:
        ln_nf = gsl_sf_result(0, 0)
        hzeta = gsl_sf_result(0, 0)
        stat_hz = gsl_sf_hzeta_e(n + 1.0, x, hzeta)
        stat_nf = gsl_sf_lnfact_e(n, ln_nf)

        ln_nf_val_IV = ln_nf.val
        ln_nf_err_IV = ln_nf.err
        hzeta_val_IV = hzeta.val
        hzeta_err_IV = hzeta.err
        stat_e = gsl_sf_exp_mult_err_e(ln_nf_val_IV, ln_nf_err_IV,
			hzeta_val_IV, hzeta_err_IV,
			result)
        if GSL_IS_EVEN(n):

            result_val_IV = result.val
            result_val = -result_val_IV
            result.val = result_val
        return GSL_ERROR_SELECT_3(stat_e, stat_nf, stat_hz)

def gsl_sf_psi_1_e(x, result):
    if x == 0.0 or x == -1.0 or x == -2.0:
        print("domain error")
    elif x > 0.0:
        return psi_n_xg0(1, x, result)
    elif x > -5.0:
        M = -floor(x)
        fx = x + M
        sum = 0.0
        if fx == 0.0:
            print("domain error")

        for m in range(0, M):
            sum += 1.0 / ((x + m)*(x + m))
        
        stat_psi = psi_n_xg0(1, fx, result)
        result_val = result.val
        result_val += sum
        result.val = result_val

        result_err = result.err
        result_err += M * GSL_DBL_EPSILON * sum
        result.err = result_err
        return stat_psi
    else:
        sin_px = sin(M_PI * x)
        d = M_PI * M_PI / (sin_px*sin_px)
        r = gsl_sf_result(0, 0)
        stat_psi = psi_n_xg0(1, 1.0 - x, r)


        r_val_IV = r.val
        result_val = d - r_val_IV
        result.val = result_val

        r_err_IV = r.err
        result_err = r_err_IV + 2.0*GSL_DBL_EPSILON*d
        result.err = result_err
        return stat_psi

def gsl_sf_psi_n_e(n, x, result):
    if n == 0:
        return gsl_sf_psi_e(x, result)
    elif n == 1:
        return gsl_sf_psi_1_e(x, result)
    elif n < 0 or x <= 0.0:
        print("domain error")
    else:
        ln_nf = gsl_sf_result(0, 0)
        hzeta = gsl_sf_result(0, 0)
        stat_hz = gsl_sf_hzeta_e(n + 1.0, x, hzeta)
        stat_nf = gsl_sf_lnfact_e(n, ln_nf)
        ln_nf_val_IV = ln_nf.val
        ln_nf_err_IV = ln_nf.err
        hzeta_val_IV = hzeta.val
        hzeta_err_IV = hzeta.err
        stat_e = gsl_sf_exp_mult_err_e(ln_nf_val_IV, ln_nf_err_IV,
			hzeta_val_IV, hzeta_err_IV,
			result)
        if GSL_IS_EVEN(n):
            result_val_IV = result.val
            result_val = -result_val_IV
            result.val = result_val
        return GSL_ERROR_SELECT_3(stat_e, stat_nf, stat_hz)

PSI_1_TABLE_NMAX = 100
psi_1_table = [0.0,  
	M_PI*M_PI / 6.0,                   
	0.644934066848226436472415,       
	0.394934066848226436472415,
	0.2838229557371153253613041,
	0.2213229557371153253613041,
	0.1813229557371153253613041,
	0.1535451779593375475835263,
	0.1331370146940314251345467,
	0.1175120146940314251345467,
	0.1051663356816857461222010,
	0.0951663356816857461222010,
	0.0869018728717683907503002,
	0.0799574284273239463058557,
	0.0740402686640103368384001,
	0.0689382278476838062261552,
	0.0644937834032393617817108,
	0.0605875334032393617817108,
	0.0571273257907826143768665,
	0.0540409060376961946237801,
	0.0512708229352031198315363,
	0.0487708229352031198315363,
	0.0465032492390579951149830,
	0.0444371335365786562720078,
	0.0425467743683366902984728,
	0.0408106632572255791873617,
	0.0392106632572255791873617,
	0.0377313733163971768204978,
	0.0363596312039143235969038,
	0.0350841209998326909438426,
	0.0338950603577399442137594,
	0.0327839492466288331026483,
	0.0317433665203020901265817,
	0.03076680402030209012658168,
	0.02984853037475571730748159,
	0.02898347847164153045627052,
	0.02816715194102928555831133,
	0.02739554700275768062003973,
	0.02666508681283803124093089,
	0.02597256603721476254286995,
	0.02531510384129102815759710,
	0.02469010384129102815759710,
	0.02409521984367056414807896,
	0.02352832641963428296894063,
	0.02298749353699501850166102,
	0.02247096461137518379091722,
	0.02197713745088135663042339,
	0.02150454765882086513703965,
	0.02105185413233829383780923,
	0.02061782635456051606003145,
	0.02020133322669712580597065,
	0.01980133322669712580597065,
	0.01941686571420193164987683,
	0.01904704322899483105816086,
	0.01869104465298913508094477,
	0.01834810912486842177504628,
	0.01801753061247172756017024,
	0.01769865306145131939690494,
	0.01739086605006319997554452,
	0.01709360088954001329302371,
	0.01680632711763538818529605,
	0.01652854933985761040751827,
	0.01625980437882562975715546,
	0.01599965869724394401313881,
	0.01574770606433893015574400,
	0.01550356543933893015574400,
	0.01526687904880638577704578,
	0.01503731063741979257227076,
	0.01481454387422086185273411,
	0.01459828089844231513993134,
	0.01438824099085987447620523,
	0.01418415935820681325171544,
	0.01398578601958352422176106,
	0.01379288478501562298719316,
	0.01360523231738567365335942,
	0.01342261726990576130858221,
	0.01324483949212798353080444,
	0.01307170929822216635628920,
	0.01290304679189732236910755,
	0.01273868124291638877278934,
	0.01257845051066194236996928,
	0.01242220051066194236996928,
	0.01226978472038606978956995,
	0.01212106372098095378719041,
	0.01197590477193174490346273,
	0.01183418141592267460867815,
	0.01169577311142440471248438,
	0.01156056489076458859566448,
	0.01142844704164317229232189,
	0.01129931481023821361463594,
	0.01117306812421372175754719,
	0.01104961133409026496742374,
	0.01092885297157366069257770,
	0.01081070552355853781923177,
	0.01069508522063334415522437,
	0.01058191183901270133041676,
	0.01047110851491297833872701,
	0.01036260157046853389428257,
	0.01025632035036012704977199,  
	0.01015219706839427948625679,  
	0.01005016666333357139524567   
]
PSI_TABLE_NMAX = 100
psi_table = [0.0,  
	-M_EULER,                          
	0.42278433509846713939348790992,  
	0.92278433509846713939348790992,
	1.25611766843180047272682124325,
	1.50611766843180047272682124325,
	1.70611766843180047272682124325,
	1.87278433509846713939348790992,
	2.01564147795560999653634505277,
	2.14064147795560999653634505277,
	2.25175258906672110764745616389,
	2.35175258906672110764745616389,
	2.44266167997581201673836525479,
	2.52599501330914535007169858813,
	2.60291809023222227314862166505,
	2.67434666166079370172005023648,
	2.74101332832746036838671690315,
	2.80351332832746036838671690315,
	2.86233685773922507426906984432,
	2.91789241329478062982462539988,
	2.97052399224214905087725697883,
	3.02052399224214905087725697883,
	3.06814303986119666992487602645,
	3.11359758531574212447033057190,
	3.15707584618530734186163491973,
	3.1987425128519740085283015864,
	3.2387425128519740085283015864,
	3.2772040513135124700667631249,
	3.3142410883505495071038001619,
	3.3499553740648352213895144476,
	3.3844381326855248765619282407,
	3.4177714660188582098952615740,
	3.4500295305349872421533260902,
	3.4812795305349872421533260902,
	3.5115825608380175451836291205,
	3.5409943255438998981248055911,
	3.5695657541153284695533770196,
	3.5973435318931062473311547974,
	3.6243705589201332743581818244,
	3.6506863483938174848844976139,
	3.6763273740348431259101386396,
	3.7013273740348431259101386396,
	3.7257176179372821503003825420,
	3.7495271417468059598241920658,
	3.7727829557002943319172153216,
	3.7955102284275670591899425943,
	3.8177324506497892814121648166,
	3.8394715810845718901078169905,
	3.8607481768292527411716467777,
	3.8815815101625860745049801110,
	3.9019896734278921969539597029,
	3.9219896734278921969539597029,
	3.9415975165651470989147440166,
	3.9608282857959163296839747858,
	3.9796962103242182164764276160,
	3.9982147288427367349949461345,
	4.0163965470245549168131279527,
	4.0342536898816977739559850956,
	4.0517975495308205809735289552,
	4.0690389288411654085597358518,
	4.0859880813835382899156680552,
	4.1026547480502049565823347218,
	4.1190481906731557762544658694,
	4.1351772229312202923834981274,
	4.1510502388042361653993711433,
	4.1666752388042361653993711433,
	4.1820598541888515500147557587,
	4.1972113693403667015299072739,
	4.2121367424746950597388624977,
	4.2268426248276362362094507330,
	4.2413353784508246420065521823,
	4.2556210927365389277208378966,
	4.2697055997787924488475984600,
	4.2835944886676813377364873489,
	4.2972931188046676391063503626,
	4.3108066323181811526198638761,
	4.3241399656515144859531972094,
	4.3372978603883565912163551041,
	4.3502848733753695782293421171,
	4.3631053861958823987421626300,
	4.3757636140439836645649474401,
	4.3882636140439836645649474401,
	4.4006092930563293435772931191,
	4.4128044150075488557724150703,
	4.4248526077786331931218126607,
	4.4367573696833950978837174226,
	4.4485220755657480390601880108,
	4.4601499825424922251066996387,
	4.4716442354160554434975042364,
	4.4830078717796918071338678728,
	4.4942438268358715824147667492,
	4.5053549379469826935258778603,
	4.5163439489359936825368668713,
	4.5272135141533849868846929582,
	4.5379662023254279976373811303,
	4.5486045001977684231692960239,
	4.5591308159872421073798223397,
	4.5695474826539087740464890064,
	4.5798567610044242379640147796,
	4.5900608426370772991885045755,
	4.6001618527380874001986055856]

def gsl_sf_psi_1_int_e(n, result):
    if n <= 0:
        print('domain error')
    elif n <= PSI_1_TABLE_NMAX:
        psi_1_table_n_IV = psi_1_table[n]
        result_val = psi_1_table_n_IV
        result.val = result_val

        result_val_IV = result.val
        result_err = GSL_DBL_EPSILON * result_val_IV
        result.err = result_err
        return GSL_SUCCESS
    else:
        c0 = -1.0 / 30.0
        c1 = 1.0 / 42.0
        c2 = -1.0 / 30.0
        ni2 = (1.0 / n)*(1.0 / n)
        ser = ni2 * ni2 * (c0 + ni2 * (c1 + c2 * ni2))

        result_val = (1.0 + 0.5 / n + 1.0 / (6.0*n*n) + ser) / n
        result.val = result_val


        result_val_IV = result.val
        result_err = GSL_DBL_EPSILON * result_val_IV
        result.err = result_err
        return GSL_SUCCESS

def gsl_sf_psi_int_e(n, result):
    if n <= 0:
        print('domain error')
    elif n <= PSI_TABLE_NMAX:
        psi_table_n_IV = psi_table[n]
        result_val = psi_table_n_IV
        result.val = result_val

        result_val_IV = result.val
        result_err = GSL_DBL_EPSILON * fabs(result_val_IV)
        result.err = result_err
        return GSL_SUCCESS
    else:
        c2 = -1.0 / 12.0
        c3 = 1.0 / 120.0
        c4 = -1.0 / 252.0
        c5 = 1.0 / 240.0
        ni2 = (1.0 / n)*(1.0 / n)
        ser = ni2 * (c2 + ni2 * (c3 + ni2 * (c4 + ni2 * c5)))
        result_val = log(n) - 0.5 / n + ser
        result.val = result_val


        result_err = GSL_DBL_EPSILON * (fabs(log(n)) + fabs(0.5 / n) + fabs(ser))
        result.err = result_err

        result_val_IV = result.val
        result_err = result.err
        result_err += GSL_DBL_EPSILON * fabs(result_val_IV)
        result.err = result_err
        return GSL_SUCCESS

def gsl_sf_lnfact_e(n, result):
    if n <= GSL_SF_FACT_NMAX:
        fact_table_n_f_IV = fact_table[n].f
        result_val = log(fact_table_n_f_IV)
        result.val = result_val

        result_val_IV = result.val
        result_err = 2.0 * GSL_DBL_EPSILON * fabs(result_val_IV)
        result.err = result_err
        return GSL_SUCCESS
    else:
        gsl_sf_lngamma_e(n + 1.0, result)
        return GSL_SUCCESS

def lngamma_sgn_sing(N, eps, lng, sgn):
    if eps == 0:
        lng.val = 0.0
        lng.val = 0.0
        sgn = 0.0
        print("edom error")
    elif N == 1:
        c0 = 0.07721566490153286061
        c1 = 0.08815966957356030521
        c2 = -0.00436125434555340577
        c3 = 0.01391065882004640689
        c4 = -0.00409427227680839100
        c5 = 0.00275661310191541584
        c6 = -0.00124162645565305019
        c7 = 0.00065267976121802783
        c8 = -0.00032205261682710437
        c9 = 0.00016229131039545456
        g5 = c5 + eps * (c6 + eps * (c7 + eps * (c8 + eps * c9)))
        g = eps * (c0 + eps * (c1 + eps * (c2 + eps * (c3 + eps * (c4 + eps * g5)))))
        gam_e = g - 1.0 - 0.5*eps*(1.0 + 3.0*eps) / (1.0 - eps * eps)


        lng_val = log(fabs(gam_e) / fabs(eps))
        lng.val = lng_val

        lng_val_IV = lng.val
        lng_err = 2.0 * GSL_DBL_EPSILON * fabs(lng_val_IV)
        lng.err = lng_err
        sgn = -1.0 if eps > 0.0 else 1.0
        return GSL_SUCCESS
    else:
        cs1 = -1.6449340668482264365
        cs2 = 0.8117424252833536436
        cs3 = -0.1907518241220842137
        cs4 = 0.0261478478176548005
        cs5 = -0.0023460810354558236
        e2 = eps * eps
        sin_ser = 1.0 + e2 * (cs1 + e2 * (cs2 + e2 * (cs3 + e2 * (cs4 + e2 * cs5))))

        aeps = fabs(eps)
        c0 = gsl_sf_result(0,0)
        psi_0 = gsl_sf_result(0,0)
        psi_1 = gsl_sf_result(0,0)
        psi_2 = gsl_sf_result(0,0)
        psi_3 = gsl_sf_result(0,0)
        psi_4 = gsl_sf_result(0,0)
        psi_5 = gsl_sf_result(0,0)
        psi_6 = gsl_sf_result(0,0)
        psi_2.val = 0.0
        psi_3.val = 0.0
        psi_4.val = 0.0
        psi_5.val = 0.0
        psi_6.val = 0.0
        gsl_sf_lnfact_e(N, c0)
        gsl_sf_psi_int_e(N + 1, psi_0)
        gsl_sf_psi_1_int_e(N + 1, psi_1)
        if aeps > 0.00001:
            gsl_sf_psi_n_e(2, N + 1.0, psi_2)
        if aeps > 0.0002:
            gsl_sf_psi_n_e(3, N + 1.0, psi_3)
        if aeps > 0.001:
            gsl_sf_psi_n_e(4, N + 1.0, psi_4)
        if aeps > 0.005:
            gsl_sf_psi_n_e(5, N + 1.0, psi_5)
        if aeps > 0.01:
            gsl_sf_psi_n_e(6, N + 1.0, psi_6)
        
        psi_0_val_IV = psi_0.val
        c1 = psi_0_val_IV
        psi_1_val_IV = psi_1.val
        c2 = psi_1_val_IV / 2.0
        psi_2_val_IV = psi_2.val
        c3 = psi_2_val_IV / 6.0
        psi_3_val_IV = psi_3.val
        c4 = psi_3_val_IV / 24.0
        psi_4_val_IV = psi_4.val
        c5 = psi_4_val_IV / 120.0
        psi_5_val_IV =  psi_5.val
        c6 = psi_5_val_IV / 720.0
        psi_6_val_IV = psi_6.val
        c7 = psi_6_val_IV / 5040.0
        c0_val_IV = c0.val
        lng_ser = c0_val_IV - eps * (c1 - eps * (c2 - eps * (c3 - eps * (c4 - eps * (c5 - eps * (c6 - eps * c7))))))
        g = -lng_ser - log(sin_ser)

        lng_val = g - log(fabs(eps))
        lng.val = lng_val

        lng_val_IV = lng.val
        c0_err = c0.err
        lng_err = c0_err + 2.0 * GSL_DBL_EPSILON * (fabs(g) + fabs(lng_val_IV))
        lng.err = lng_err
        sgn = (-1.0 if GSL_IS_ODD(N) else 1.0) *(1.0 if eps > 0.0 else -1.0)
        return GSL_SUCCESS

def lngamma_1_pade(eps, result):
    n1 = -1.0017419282349508699871138440
    n2 = 1.7364839209922879823280541733
    d1 = 1.2433006018858751556055436011
    d2 = 5.0456274100274010152489597514
    num = (eps + n1) * (eps + n2)
    den = (eps + d1) * (eps + d2)
    pade = 2.0816265188662692474880210318 * num / den
    c0 = 0.004785324257581753
    c1 = -0.01192457083645441
    c2 = 0.01931961413960498
    c3 = -0.02594027398725020
    c4 = 0.03141928755021455
    eps5 = eps * eps*eps*eps*eps
    corr = eps5 * (c0 + eps * (c1 + eps * (c2 + eps * (c3 + c4 * eps))))

    result_val = eps * (pade + corr)
    result.val = result_val

    result_val_IV =result.val
    result_err = 2.0 * GSL_DBL_EPSILON * fabs(result_val_IV)
    result.err = result_err
    return GSL_SUCCESS

def lngamma_2_pade(eps, result):
    n1 = 1.000895834786669227164446568
    n2 = 4.209376735287755081642901277
    d1 = 2.618851904903217274682578255
    d2 = 10.85766559900983515322922936
    num = (eps + n1) * (eps + n2)
    den = (eps + d1) * (eps + d2)
    pade = 2.85337998765781918463568869 * num / den
    c0 = 0.0001139406357036744
    c1 = -0.0001365435269792533
    c2 = 0.0001067287169183665
    c3 = -0.0000693271800931282
    c4 = 0.0000407220927867950
    eps5 = eps * eps*eps*eps*eps
    corr = eps5 * (c0 + eps * (c1 + eps * (c2 + eps * (c3 + c4 * eps))))

    result_val = eps * (pade + corr)
    result.val = result_val

    result_val_IV = result.val
    result_err = 2.0 * GSL_DBL_EPSILON * fabs(result_val_IV)
    result.err = result_err
    return GSL_SUCCESS

lanczos_7_c = [0.99999999999980993227684700473478,
	676.520368121885098567009190444019,
	-1259.13921672240287047156078755283,
	771.3234287776530788486528258894,
	-176.61502916214059906584551354,
	12.507343278686904814458936853,
	-0.13857109526572011689554707,
	9.984369578019570859563e-6,
	1.50563273514931155834e-7]

def lngamma_lanczos(x, result):

    x -= 1.0

    Ag = lanczos_7_c[0]
    for k in range(1,9):
        lanczos_7_c_k_IV = lanczos_7_c[k]
        Ag += lanczos_7_c_k_IV / (x + k)
    term1 = (x + 0.5)*log((x + 7.5) / M_E)
    term2 = LogRootTwoPi_ + log(Ag)

    result_val = term1 + (term2 - 7.0)
    result.val = result_val

    result_err = 2.0 * GSL_DBL_EPSILON * (fabs(term1) + fabs(term2) + 7.0)
    result.err = result_err

    result_val_IV = result.val
    result_err = result.err
    result_err += GSL_DBL_EPSILON * fabs(result_val_IV)
    result.err = result_err
    return GSL_SUCCESS

def lngamma_sgn_0(eps, lgn, sgn):
    c1 = -0.07721566490153286061
    c2 = -0.01094400467202744461
    c3 = 0.09252092391911371098
    c4 = -0.01827191316559981266
    c5 = 0.01800493109685479790
    c6 = -0.00685088537872380685
    c7 = 0.00399823955756846603
    c8 = -0.00189430621687107802
    c9 = 0.00097473237804513221
    c10 = -0.00048434392722255893
    g6 = c6 + eps * (c7 + eps * (c8 + eps * (c9 + eps * c10)))
    g = eps * (c1 + eps * (c2 + eps * (c3 + eps * (c4 + eps * (c5 + eps * g6)))))

    gee = g + 1.0 / (1.0 + eps) + 0.5*eps

    
    lgn_val = log(gee / fabs(eps))
    lgn.val = lgn_val

    lgn_val_IV = lgn.val
    lgn_err = 4.0 * GSL_DBL_EPSILON * fabs(lgn_val_IV)
    lgn.err = lgn_err

    sgn = GSL_SIGN(eps)

    return GSL_SUCCESS

def gsl_sf_lngamma_e(x, result):
    if fabs(x - 1.0) < 0.01:
        stat = lngamma_1_pade(x - 1.0, result)
        result_err = result.err
        result_err *= 1.0 / (GSL_DBL_EPSILON + fabs(x - 1.0))
        result.err = result_err
        
        return stat

    elif fabs(x - 2.0) < 0.01:
        stat = lngamma_2_pade(x - 2.0, result)
        result_err = result.err
        result_err *= 1.0 / (GSL_DBL_EPSILON + fabs(x - 2.0))
        result.err = result_err
        return stat
    
    elif x >= 0.5:
        return_val = lngamma_lanczos(x, result)
        return return_val

    elif x == 0.0:
        print("domain error")

    elif fabs(x) < 0.02:
        sgn = 0.0
        return_val = lngamma_sgn_0(x, result, sgn)
        return return_val

    elif x > -0.5 / (GSL_DBL_EPSILON*M_PI):
        z = 1.0 - x
        s = sin(M_PI*z)
        _as = fabs(s)
        if s == 0.0:
            print("domain error")
        
        elif _as < M_PI*0.015:

            if x < float('-inf') + 2.0:

                result_val = 0.0
                result.val = result_val
                result_err = 0.0
                result.err = result_err

                print("eround error")
            else:
                N = -(x - 0.5)
                eps = x + N
                sgn = 0
                return_val = lngamma_sgn_sing(N, eps, result, sgn)
                return return_val
        else:
            lg_z = gsl_sf_result(0,0)
            lngamma_lanczos(z, lg_z)

            lg_z_val_IV = lg_z.val
            result_val = M_LNPI - (log(_as) + lg_z_val_IV)
            result.val = result_val

            result_val_IV = result.val
            lg_z_err = lg_z.err
            result_err = 2.0 * GSL_DBL_EPSILON * fabs(result_val_IV) + lg_z_err
            result.err = result_err

            return GSL_SUCCESS
    else:
        result_val = 0.0
        result.val = result_val
        result_err = 0.0
        result.err = result_err
        print("eround error")

def gsl_sf_lngamma(x):
    result = gsl_sf_result(0.0, 0.0)
    return EVAL_RESULT(gsl_sf_lngamma_e(x, result), result)


good_dict = {}
cnt = 0


for i in np.arange(0.01, 10.01, 0.01):
    good_dict[cnt] = gsl_sf_lngamma(i)
    cnt += 1










