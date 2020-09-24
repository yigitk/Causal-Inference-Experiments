import numpy as np
import pandas as pd
import numbers
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from phi import *
import random
import os
from math import *
from collections import namedtuple


from lngammaLong import good_dict
os.system('python lngammaLong.py') 

from phi import *
reason_0=None;pade_0=None;pade_1=None;cs_c_cs_order_IV_0=None;delta_1=None;delta_0=None;delta_2=None;delta_3=None;delta_4=None;aeps_0=None;aeps_1=None;den_0=None;den_1=None;result_val_0=None;result_val_1=None;result_val_2=None;result_val_3=None;result_val_4=None;result_val_5=None;result_val_6=None;result_val_7=None;result_val_8=None;result_val_9=None;result_val_10=None;result_val_11=None;result_val_12=None;result_val_13=None;result_val_14=None;result_val_15=None;result_val_16=None;result_val_17=None;result_val_18=None;result_val_19=None;result_val_20=None;result_val_21=None;result_val_22=None;result_val_23=None;result_val_24=None;result_val_25=None;result_val_26=None;result_val_27=None;result_val_28=None;result_val_29=None;result_val_30=None;result_val_31=None;result_val_32=None;result_val_33=None;result_val_34=None;result_val_35=None;result_val_36=None;result_val_37=None;result_val_38=None;result_val_39=None;result_val_40=None;result_val_41=None;result_val_42=None;result_val_43=None;result_val_44=None;result_val_45=None;result_val_46=None;result_val_47=None;lanczos_7_c_k_IV_1=None;lanczos_7_c_k_IV_0=None;lanczos_7_c_k_IV_2=None;hzeta_c_j_1_IV_1=None;hzeta_c_j_1_IV_0=None;hzeta_c_j_1_IV_2=None;hzeta_c_j_1_IV_3=None;hzeta_c_j_1_IV_4=None;ln_nf_val_IV_0=None;ln_nf_val_IV_1=None;ln_nf_val_IV_2=None;ln_nf_val_IV_3=None;order_0=None;val_0=None;gam_e_0=None;gam_e_1=None;scp_0=None;scp_2=None;scp_1=None;scp_3=None;scp_4=None;scp_5=None;hzeta_val_IV_0=None;hzeta_val_IV_1=None;hzeta_val_IV_2=None;hzeta_val_IV_3=None;eps5_0=None;eps5_1=None;lng_0=None;eps_0=None;eps_1=None;eps_2=None;eps_3=None;eps_4=None;eps_5=None;eps_6=None;eps_7=None;lgn_val_0=None;kmax_0=None;kmax_1=None;kmax_2=None;return_val_0=None;return_val_1=None;return_val_2=None;return_val_3=None;return_val_4=None;return_val_5=None;c0_0=None;c0_1=None;c0_2=None;c0_3=None;c0_4=None;c0_5=None;c0_6=None;c1_0=None;c1_1=None;c1_2=None;c1_3=None;c1_4=None;c1_5=None;c1_6=None;c1_7=None;fact_table_n_f_IV_0=None;fact_table_n_f_IV_1=None;c2_0=None;c2_1=None;c2_2=None;c2_3=None;c2_4=None;c2_5=None;c2_6=None;c2_7=None;c2_8=None;c2_9=None;c3_0=None;c3_1=None;c3_2=None;c3_3=None;c3_4=None;c3_5=None;c3_6=None;c3_7=None;lnr_0=None;lnr_1=None;c4_0=None;c4_1=None;c4_2=None;c4_3=None;c4_4=None;c4_5=None;c4_6=None;c4_7=None;c5_0=None;c5_1=None;c5_2=None;c5_3=None;c5_4=None;c5_5=None;c6_0=None;c6_1=None;c6_2=None;c6_3=None;c7_0=None;c7_1=None;c7_2=None;c7_3=None;lng_val_IV_0=None;lng_val_IV_1=None;lng_val_IV_2=None;c8_0=None;c8_1=None;c8_2=None;stat_hz_0=None;stat_hz_1=None;stat_hz_2=None;stat_hz_3=None;c9_0=None;c9_1=None;c9_2=None;cs_b_IV_0=None;ay_0=None;result_c_0=None;result_c_1=None;result_c_2=None;t1_0=None;t1_1=None;t1_2=None;t1_3=None;t1_4=None;t2_0=None;t2_1=None;t2_2=None;t2_3=None;status_0=None;t3_0=None;t3_1=None;t3_2=None;r_err_IV_0=None;r_err_IV_1=None;psi_1_table_n_IV_0=None;psi_1_table_n_IV_1=None;psi_6_val_IV_0=None;psi_6_val_IV_1=None;psi_5_val_IV_0=None;psi_5_val_IV_1=None;result_c_err_IV_0=None;result_c_err_IV_1=None;result_c_err_IV_2=None;result_c_err_IV_3=None;result_c_err_IV_4=None;result_c_err_IV_5=None;result_c_err_IV_6=None;result_c_err_IV_7=None;result_c_err_IV_8=None;Ag_0=None;Ag_2=None;Ag_1=None;Ag_3=None;gee_0=None;cs_c_j_IV_1=None;cs_c_j_IV_0=None;cs_c_j_IV_2=None;eab_0=None;eab_1=None;eab_2=None;sum_0=None;sum_2=None;sum_1=None;sum_3=None;sum_4=None;max_bits_0=None;max_bits_1=None;M_0=None;M_1=None;M_2=None;M_3=None;M_4=None;d1_0=None;d1_1=None;N_0=None;N_1=None;N_2=None;N_3=None;N_4=None;N_5=None;N_6=None;N_7=None;d2_0=None;d2_1=None;result_0=None;result_1=None;result_2=None;result_3=None;result_4=None;result_5=None;result_6=None;result_7=None;result_8=None;result_9=None;result_10=None;result_11=None;result_12=None;result_13=None;result_14=None;result_15=None;result_16=None;lgn_0=None;gsl_errno_0=None;psi_2_val_IV_0=None;psi_2_val_IV_1=None;c0_val_IV_0=None;c0_val_IV_1=None;psi_1_val_IV_0=None;psi_1_val_IV_1=None;a_0=None;a_1=None;a_2=None;a_3=None;a_4=None;a_5=None;ser_0=None;ser_1=None;ser_2=None;ser_3=None;b_0=None;b_1=None;b_2=None;b_3=None;b_4=None;b_5=None;stat_0=None;stat_1=None;stat_2=None;c_0=None;c_1=None;c_2=None;c_3=None;c_4=None;err_0=None;d_0=None;d_2=None;d_1=None;d_3=None;d_4=None;d_5=None;d_6=None;e_0=None;e_2=None;e_1=None;e_3=None;e_4=None;sy_0=None;sy_1=None;sy_2=None;g_0=None;g_1=None;g_2=None;g_3=None;lgn_err_0=None;stat_nf_0=None;stat_nf_1=None;stat_nf_2=None;stat_nf_3=None;j_0=None;j_1=None;k_0=None;k_1=None;m_0=None;e2_0=None;e2_1=None;n_0=None;n_1=None;n_2=None;n_3=None;n_4=None;n_5=None;n_6=None;cs_0=None;q_0=None;r_0=None;r_1=None;s_0=None;s_1=None;s_2=None;s_3=None;s_4=None;s_5=None;t_0=None;t_1=None;v_0=None;v_1=None;v_2=None;v_3=None;v_4=None;x_0=None;x_1=None;x_2=None;x_3=None;x_4=None;x_5=None;x_6=None;x_7=None;x_8=None;x_9=None;x_10=None;x_11=None;psi_table_n_IV_0=None;psi_table_n_IV_1=None;y_0=None;y_1=None;y_2=None;z_0=None;z_1=None;order_sp_0=None;dd_0=None;dd_2=None;dd_1=None;dd_3=None;corr_0=None;corr_1=None;n1_0=None;n1_1=None;n2_0=None;n2_1=None;lg_z_0=None;lg_z_1=None;lg_z_2=None;num_0=None;num_1=None;jmax_0=None;jmax_1=None;jmax_2=None;ly_0=None;ly_1=None;c0_err_0=None;c0_err_1=None;cs_a_IV_0=None;dx_0=None;result_val_IV_0=None;result_val_IV_1=None;result_val_IV_2=None;result_val_IV_3=None;result_val_IV_4=None;result_val_IV_5=None;result_val_IV_6=None;result_val_IV_7=None;result_val_IV_8=None;result_val_IV_9=None;result_val_IV_10=None;result_val_IV_11=None;result_val_IV_12=None;result_val_IV_13=None;result_val_IV_14=None;result_val_IV_15=None;result_val_IV_16=None;result_val_IV_17=None;result_val_IV_18=None;result_val_IV_19=None;result_val_IV_20=None;result_val_IV_21=None;result_val_IV_22=None;result_val_IV_23=None;result_val_IV_24=None;result_val_IV_25=None;result_val_IV_26=None;result_val_IV_27=None;result_val_IV_28=None;result_val_IV_29=None;result_val_IV_30=None;result_val_IV_31=None;result_val_IV_32=None;result_val_IV_33=None;result_val_IV_34=None;dy_0=None;psi_0_0=None;psi_0_1=None;psi_1_0=None;psi_1_1=None;psi_2_0=None;psi_2_1=None;result_c_val_IV_0=None;result_c_val_IV_1=None;result_c_val_IV_2=None;result_c_val_IV_3=None;result_c_val_IV_4=None;result_c_val_IV_5=None;result_c_val_IV_6=None;result_c_val_IV_7=None;result_c_val_IV_8=None;psi_3_0=None;psi_3_1=None;sgn_0=None;sgn_1=None;sgn_2=None;sgn_3=None;sgn_4=None;sgn_5=None;sgn_6=None;sgn_7=None;sgn_8=None;sgn_9=None;sgn_10=None;sgn_11=None;psi_4_0=None;psi_4_1=None;psi_5_0=None;psi_5_1=None;hzeta_err_IV_0=None;hzeta_err_IV_1=None;hzeta_err_IV_2=None;hzeta_err_IV_3=None;psi_6_0=None;psi_6_1=None;lg_z_val_IV_0=None;lg_z_val_IV_1=None;lg_z_val_IV_2=None;lng_val_0=None;lng_val_1=None;lng_val_2=None;result_err_0=None;result_err_1=None;result_err_2=None;result_err_3=None;result_err_4=None;result_err_5=None;result_err_6=None;result_err_7=None;result_err_8=None;result_err_9=None;result_err_10=None;result_err_11=None;result_err_12=None;result_err_13=None;result_err_14=None;result_err_15=None;result_err_16=None;result_err_17=None;result_err_18=None;result_err_19=None;result_err_20=None;result_err_21=None;result_err_22=None;result_err_23=None;result_err_24=None;result_err_25=None;result_err_26=None;result_err_27=None;result_err_28=None;result_err_29=None;result_err_30=None;result_err_31=None;result_err_32=None;result_err_33=None;result_err_34=None;result_err_35=None;result_err_36=None;result_err_37=None;result_err_38=None;result_err_39=None;result_err_40=None;result_err_41=None;result_err_42=None;result_err_43=None;result_err_44=None;result_err_45=None;result_err_46=None;result_err_47=None;result_err_48=None;result_err_49=None;result_err_50=None;result_err_51=None;result_err_52=None;result_err_53=None;result_err_54=None;result_err_55=None;result_err_56=None;result_err_57=None;result_err_58=None;result_err_59=None;result_err_60=None;result_err_61=None;result_err_62=None;result_err_63=None;result_err_64=None;result_err_65=None;result_err_66=None;result_err_67=None;result_err_68=None;result_err_69=None;result_err_70=None;result_err_71=None;result_err_72=None;sin_ser_0=None;sin_ser_1=None;g5_0=None;g5_1=None;g6_0=None;ln_nf_0=None;ln_nf_1=None;ln_nf_2=None;ln_nf_3=None;hzeta_0=None;hzeta_1=None;hzeta_2=None;hzeta_3=None;ex_0=None;ex_1=None;pmax_0=None;pmax_1=None;pmax_2=None;cs1_0=None;cs1_1=None;p1_0=None;p1_1=None;p1_2=None;p2_0=None;p2_1=None;p2_2=None;cs3_0=None;cs3_1=None;p3_0=None;p3_1=None;p3_2=None;cs2_0=None;cs2_1=None;c10_0=None;cs5_0=None;cs5_1=None;psi_4_val_IV_0=None;psi_4_val_IV_1=None;cs4_0=None;cs4_1=None;eMN_0=None;eMN_1=None;eMN_2=None;fn_0=None;_as_0=None;_as_1=None;fx_0=None;fx_1=None;stat_psi_0=None;stat_psi_1=None;stat_psi_2=None;term2_0=None;psi_0_val_IV_0=None;psi_0_val_IV_1=None;ni2_0=None;ni2_1=None;ni2_2=None;ni2_3=None;term1_0=None;y2_0=None;psi_3_val_IV_0=None;psi_3_val_IV_1=None;ln_term0_0=None;ln_term0_1=None;pcp_0=None;pcp_2=None;pcp_1=None;pcp_3=None;pcp_4=None;pcp_5=None;value_0=None;temp_1=None;temp_0=None;temp_2=None;temp_3=None;r_val_IV_0=None;r_val_IV_1=None;lng_err_0=None;lng_err_1=None;lng_err_2=None;ans_0=None;ans_2=None;ans_1=None;ans_3=None;ans_5=None;ans_4=None;ans_6=None;ans_7=None;ans_8=None;cs_c_0_IV_0=None;lng_ser_0=None;lng_ser_1=None;ln_nf_err_IV_0=None;ln_nf_err_IV_1=None;ln_nf_err_IV_2=None;ln_nf_err_IV_3=None;lgn_val_IV_0=None;stat_e_0=None;stat_e_1=None;stat_e_2=None;stat_e_3=None;sin_px_0=None;sin_px_1=None;lg_z_err_0=None;lg_z_err_1=None;lg_z_err_2=None

M_PI=3.14159265358979323846264338328 
M_LNPI=1.14472988584940017414342735135 
GSL_DBL_EPSILON=2.2204460492503131e-16 
M_E=2.71828182845904523536028747135 
LogRootTwoPi_=0.9189385332046727418 
M_EULER=0.57721566490153286060651209008 
GSL_LOG_DBL_MIN=(-7.0839641853226408e+02) 
GSL_LOG_DBL_MAX=7.0978271289338397e+02 
GSL_SQRT_DBL_MAX=1.3407807929942596e+154 
GSL_SQRT_DBL_MIN=1.4916681462400413e-154 
GSL_SF_FACT_NMAX=170 
def GSL_IS_ODD(n):
    n_0 = n;
    

    return n_0%1==1

def GSL_IS_EVEN(n):
    n_1 = n;
    

    return n_1%1==0

def GSL_SIGN(x):
    x_0 = x;
    

    if x_0>=0.0:
        return 1
    return 0

def GSL_ERROR_SELECT_2(a,b):
    a_0 = a;b_0 = b;
    

    return a_0 if a_0!=GSL_SUCCESS else (b_0 if b_0!=GSL_SUCCESS else GSL_SUCCESS)

def GSL_ERROR_SELECT_3(a,b,c):
    a_1 = a;b_1 = b;c_0 = c;
    

    return a_1 if a_1!=GSL_SUCCESS else GSL_ERROR_SELECT_2(b_1,c_0)

GSL_SUCCESS=0 
GSL_FAILURE=-1 
GSL_CONTINUE=-2 
GSL_EDOM=1 
GSL_ERANGE=2 
GSL_EFAULT=3 
GSL_EINVAL=4 
GSL_EFAILED=5 
GSL_EFACTOR=6 
GSL_ESANITY=7 
GSL_ENOMEM=8 
GSL_EBADFUNC=9 
GSL_ERUNAWAY=1 
GSL_EMAXITER=11 
GSL_EZERODIV=12 
GSL_EBADTOL=13 
GSL_ETOL=14 
GSL_EUNDRFLW=15 
GSL_EOVRFLW=16 
GSL_ELOSS=17 
GSL_EROUND=18 
GSL_EBADLEN=19 
GSL_ENOTSQR=20 
GSL_ESING=21 
GSL_EDIVERGE=22 
GSL_EUNSUP=23 
GSL_EUNIMPL=24 
GSL_ECACHE=25 
GSL_ETABLE=26 
GSL_ENOPROG=27 
GSL_ENOPROGJ=28 
GSL_ETOLF=29 
GSL_ETOLX=30 
GSL_ETOLG=31 
GSL_EOF=32 
class cheb_series:
    def __init__(self,c,order,a,b,order_sp):
        c_1 = c;order_0 = order;a_2 = a;b_2 = b;order_sp_0 = order_sp;
        

        self.c=c_1 
        self.order=order_0 
        self.a=a_2 
        self.b=b_2 
        self.order_sp=order_sp_0 

FactStruct=namedtuple("FactStruct","n f i") 
fact_table=[FactStruct(0,1.0,1),FactStruct(1,1.0,1),FactStruct(2,2.0,2),FactStruct(3,6.0,6),FactStruct(4,24.0,24),FactStruct(5,120.0,120),FactStruct(6,720.0,720),FactStruct(7,5040.0,5040),FactStruct(8,40320.0,40320),FactStruct(9,362880.0,362880),FactStruct(10,3628800.0,3628800),FactStruct(11,39916800.0,39916800),FactStruct(12,479001600.0,479001600),FactStruct(13,6227020800.0,0),FactStruct(14,87178291200.0,0),FactStruct(15,1307674368000.0,0),FactStruct(16,20922789888000.0,0),FactStruct(17,355687428096000.0,0),FactStruct(18,6402373705728000.0,0),FactStruct(19,121645100408832000.0,0),FactStruct(20,2432902008176640000.0,0),FactStruct(21,51090942171709440000.0,0),FactStruct(22,1124000727777607680000.0,0),FactStruct(23,25852016738884976640000.0,0),FactStruct(24,620448401733239439360000.0,0),FactStruct(25,15511210043330985984000000.0,0),FactStruct(26,403291461126605635584000000.0,0),FactStruct(27,10888869450418352160768000000.0,0),FactStruct(28,304888344611713860501504000000.0,0),FactStruct(29,8841761993739701954543616000000.0,0),FactStruct(30,265252859812191058636308480000000.0,0),FactStruct(31,8222838654177922817725562880000000.0,0),FactStruct(32,263130836933693530167218012160000000.0,0),FactStruct(33,8683317618811886495518194401280000000.0,0),FactStruct(34,2.95232799039604140847618609644e38,0),FactStruct(35,1.03331479663861449296666513375e40,0),FactStruct(36,3.71993326789901217467999448151e41,0),FactStruct(37,1.37637530912263450463159795816e43,0),FactStruct(38,5.23022617466601111760007224100e44,0),FactStruct(39,2.03978820811974433586402817399e46,0),FactStruct(40,8.15915283247897734345611269600e47,0),FactStruct(41,3.34525266131638071081700620534e49,0),FactStruct(42,1.40500611775287989854314260624e51,0),FactStruct(43,6.04152630633738356373551320685e52,0),FactStruct(44,2.65827157478844876804362581101e54,0),FactStruct(45,1.19622220865480194561963161496e56,0),FactStruct(46,5.50262215981208894985030542880e57,0),FactStruct(47,2.58623241511168180642964355154e59,0),FactStruct(48,1.24139155925360726708622890474e61,0),FactStruct(49,6.08281864034267560872252163321e62,0),FactStruct(50,3.04140932017133780436126081661e64,0),FactStruct(51,1.55111875328738228022424301647e66,0),FactStruct(52,8.06581751709438785716606368564e67,0),FactStruct(53,4.27488328406002556429801375339e69,0),FactStruct(54,2.30843697339241380472092742683e71,0),FactStruct(55,1.26964033536582759259651008476e73,0),FactStruct(56,7.10998587804863451854045647464e74,0),FactStruct(57,4.05269195048772167556806019054e76,0),FactStruct(58,2.35056133128287857182947491052e78,0),FactStruct(59,1.38683118545689835737939019720e80,0),FactStruct(60,8.32098711274139014427634118320e81,0),FactStruct(61,5.07580213877224798800856812177e83,0),FactStruct(62,3.14699732603879375256531223550e85,0),FactStruct(63,1.982608315404440064116146708360e87,0),FactStruct(64,1.268869321858841641034333893350e89,0),FactStruct(65,8.247650592082470666723170306800e90,0),FactStruct(66,5.443449390774430640037292402480e92,0),FactStruct(67,3.647111091818868528824985909660e94,0),FactStruct(68,2.480035542436830599600990418570e96,0),FactStruct(69,1.711224524281413113724683388810e98,0),FactStruct(70,1.197857166996989179607278372170e100,0),FactStruct(71,8.504785885678623175211676442400e101,0),FactStruct(72,6.123445837688608686152407038530e103,0),FactStruct(73,4.470115461512684340891257138130e105,0),FactStruct(74,3.307885441519386412259530282210e107,0),FactStruct(75,2.480914081139539809194647711660e109,0),FactStruct(76,1.885494701666050254987932260860e111,0),FactStruct(77,1.451830920282858696340707840860e113,0),FactStruct(78,1.132428117820629783145752115870e115,0),FactStruct(79,8.946182130782975286851441715400e116,0),FactStruct(80,7.156945704626380229481153372320e118,0),FactStruct(81,5.797126020747367985879734231580e120,0),FactStruct(82,4.753643337012841748421382069890e122,0),FactStruct(83,3.945523969720658651189747118010e124,0),FactStruct(84,3.314240134565353266999387579130e126,0),FactStruct(85,2.817104114380550276949479442260e128,0),FactStruct(86,2.422709538367273238176552320340e130,0),FactStruct(87,2.107757298379527717213600518700e132,0),FactStruct(88,1.854826422573984391147968456460e134,0),FactStruct(89,1.650795516090846108121691926250e136,0),FactStruct(90,1.485715964481761497309522733620e138,0),FactStruct(91,1.352001527678402962551665687590e140,0),FactStruct(92,1.243841405464130725547532432590e142,0),FactStruct(93,1.156772507081641574759205162310e144,0),FactStruct(94,1.087366156656743080273652852570e146,0),FactStruct(95,1.032997848823905926259970209940e148,0),FactStruct(96,9.916779348709496892095714015400e149,0),FactStruct(97,9.619275968248211985332842594960e151,0),FactStruct(98,9.426890448883247745626185743100e153,0),FactStruct(99,9.332621544394415268169923885600e155,0),FactStruct(100,9.33262154439441526816992388563e157,0),FactStruct(101,9.42594775983835942085162312450e159,0),FactStruct(102,9.61446671503512660926865558700e161,0),FactStruct(103,9.90290071648618040754671525458e163,0),FactStruct(104,1.02990167451456276238485838648e166,0),FactStruct(105,1.08139675824029090050410130580e168,0),FactStruct(106,1.146280563734708354534347384148e170,0),FactStruct(107,1.226520203196137939351751701040e172,0),FactStruct(108,1.324641819451828974499891837120e174,0),FactStruct(109,1.443859583202493582204882102460e176,0),FactStruct(110,1.588245541522742940425370312710e178,0),FactStruct(111,1.762952551090244663872161047110e180,0),FactStruct(112,1.974506857221074023536820372760e182,0),FactStruct(113,2.231192748659813646596607021220e184,0),FactStruct(114,2.543559733472187557120132004190e186,0),FactStruct(115,2.925093693493015690688151804820e188,0),FactStruct(116,3.393108684451898201198256093590e190,0),FactStruct(117,3.96993716080872089540195962950e192,0),FactStruct(118,4.68452584975429065657431236281e194,0),FactStruct(119,5.57458576120760588132343171174e196,0),FactStruct(120,6.68950291344912705758811805409e198,0),FactStruct(121,8.09429852527344373968162284545e200,0),FactStruct(122,9.87504420083360136241157987140e202,0),FactStruct(123,1.21463043670253296757662432419e205,0),FactStruct(124,1.50614174151114087979501416199e207,0),FactStruct(125,1.88267717688892609974376770249e209,0),FactStruct(126,2.37217324288004688567714730514e211,0),FactStruct(127,3.01266001845765954480997707753e213,0),FactStruct(128,3.85620482362580421735677065923e215,0),FactStruct(129,4.97450422247728744039023415041e217,0),FactStruct(130,6.46685548922047367250730439554e219,0),FactStruct(131,8.47158069087882051098456875820e221,0),FactStruct(132,1.11824865119600430744996307608e224,0),FactStruct(133,1.48727070609068572890845089118e226,0),FactStruct(134,1.99294274616151887673732419418e228,0),FactStruct(135,2.69047270731805048359538766215e230,0),FactStruct(136,3.65904288195254865768972722052e232,0),FactStruct(137,5.01288874827499166103492629211e234,0),FactStruct(138,6.91778647261948849222819828311e236,0),FactStruct(139,9.61572319694108900419719561353e238,0),FactStruct(140,1.34620124757175246058760738589e241,0),FactStruct(141,1.89814375907617096942852641411e243,0),FactStruct(142,2.69536413788816277658850750804e245,0),FactStruct(143,3.85437071718007277052156573649e247,0),FactStruct(144,5.55029383273930478955105466055e249,0),FactStruct(145,8.04792605747199194484902925780e251,0),FactStruct(146,1.17499720439091082394795827164e254,0),FactStruct(147,1.72724589045463891120349865931e256,0),FactStruct(148,2.55632391787286558858117801578e258,0),FactStruct(149,3.80892263763056972698595524351e260,0),FactStruct(150,5.71338395644585459047893286526e262,0),FactStruct(151,8.62720977423324043162318862650e264,0),FactStruct(152,1.31133588568345254560672467123e267,0),FactStruct(153,2.00634390509568239477828874699e269,0),FactStruct(154,3.08976961384735088795856467036e271,0),FactStruct(155,4.78914290146339387633577523906e273,0),FactStruct(156,7.47106292628289444708380937294e275,0),FactStruct(157,1.17295687942641442819215807155e278,0),FactStruct(158,1.85327186949373479654360975305e280,0),FactStruct(159,2.94670227249503832650433950735e282,0),FactStruct(160,4.71472363599206132240694321176e284,0),FactStruct(161,7.59070505394721872907517857094e286,0),FactStruct(162,1.22969421873944943411017892849e289,0),FactStruct(163,2.00440157654530257759959165344e291,0),FactStruct(164,3.28721858553429622726333031164e293,0),FactStruct(165,5.42391066613158877498449501421e295,0),FactStruct(166,9.00369170577843736647426172359e297,0),FactStruct(167,1.50361651486499904020120170784e300,0),FactStruct(168,2.52607574497319838753801886917e302,0),FactStruct(169,4.26906800900470527493925188890e304,0),FactStruct(170,7.25741561530799896739672821113e306,0)] 
class gsl_sf_result:
    def __init__(self,val,err):
        val_0 = val;err_0 = err;
        

        self.val=val_0 
        self.err=err_0 

def GSL_ERROR_VAL(reason,gsl_errno,value):
    reason_0 = reason;gsl_errno_0 = gsl_errno;value_0 = value;
    

    return 

def EVAL_RESULT(fn,result):
    fn_0 = fn;result_0 = result;
    result_val_0=None;status_0=None;

    status_0=fn_0 
    if status_0!=GSL_SUCCESS:
        GSL_ERROR_VAL(fn_0,status_0,result_0.val) 
    result_val_0=result_0.val 
    lo = locals()
    record_locals(lo, test_counter) 
    return result_val_0

def cheb_eval_e(cs,x,result):
    cs_0 = cs;x_1 = x;result_1 = result;
    dd_0=None;dd_2=None;dd_1=None;dd_3=None;temp_1=None;temp_0=None;temp_2=None;temp_3=None;d_0=None;d_2=None;d_1=None;d_3=None;d_4=None;e_0=None;e_2=None;e_1=None;e_3=None;e_4=None;cs_c_j_IV_1=None;cs_c_j_IV_0=None;cs_c_j_IV_2=None;cs_c_cs_order_IV_0=None;cs_c_0_IV_0=None;result_err_0=None;result_val_1=None;cs_a_IV_0=None;cs_b_IV_0=None;y_0=None;y2_0=None;

    d_0=0.0 
    dd_0=0.0 
    cs_a_IV_0=cs_0.a 
    cs_b_IV_0=cs_0.b 
    y_0=(2.0*x_1-cs_a_IV_0-cs_b_IV_0)/(cs_b_IV_0-cs_a_IV_0) 
    y2_0=2.0*y_0 
    e_0=0.0 
    phi0 = Phi()
    for j_0 in range(cs_0.order,0,-1):
        phi0.set()
        dd_2 = phi0.phiEntry(dd_0,dd_1)
        temp_1 = phi0.phiEntry(None,temp_0)
        d_2 = phi0.phiEntry(d_0,d_1)
        e_2 = phi0.phiEntry(e_0,e_1)
        cs_c_j_IV_1 = phi0.phiEntry(None,cs_c_j_IV_0)

        temp_0=d_2 
        cs_c_j_IV_0=cs_0.c[j_0] 
        d_1=y2_0*d_2-dd_2+cs_c_j_IV_0 
        e_1 = e_2+fabs(y2_0*temp_0)+fabs(dd_2)+fabs(cs_c_j_IV_0)
        dd_1=temp_0 
    dd_3 = phi0.phiExit(dd_0,dd_1)
    temp_2 = phi0.phiExit(None,temp_0)
    d_3 = phi0.phiExit(d_0,d_1)
    e_3 = phi0.phiExit(e_0,e_1)
    cs_c_j_IV_2 = phi0.phiExit(None,cs_c_j_IV_0)
    temp_3=d_3 
    cs_c_0_IV_0=cs_0.c[0] 
    d_4=y_0*d_3-dd_3+0.5*cs_c_0_IV_0 
    e_4 = e_3+fabs(y_0*temp_3)+fabs(dd_3)+0.5*fabs(cs_c_0_IV_0)
    result_val_1=d_4 
    result_1.val=result_val_1 
    cs_c_cs_order_IV_0=cs_0.c[cs_0.order] 
    result_err_0=GSL_DBL_EPSILON*e_4+fabs(cs_c_cs_order_IV_0) 
    result_1.err=result_err_0
    lo = locals()
    record_locals(lo, test_counter)  
    return GSL_SUCCESS

psics_data=[-.038057080835217922,.491415393029387130,-.056815747821244730,.008357821225914313,-.001333232857994342,.000220313287069308,-.000037040238178456,.000006283793654854,-.000001071263908506,.000000183128394654,-.000000031353509361,.000000005372808776,-.000000000921168141,.000000000157981265,-.000000000027098646,.000000000004648722,-.000000000000797527,.000000000000136827,-.000000000000023475,.000000000000004027,-.000000000000000691,.000000000000000118,-.000000000000000020] 
psi_cs=cheb_series(psics_data,22,-1,1,17) 
apsics_data=[-.0204749044678185,-.0101801271534859,.0000559718725387,-.0000012917176570,.0000000572858606,-.0000000038213539,.0000000003397434,-.0000000000374838,.0000000000048990,-.0000000000007344,.0000000000001233,-.0000000000000228,.0000000000000045,-.0000000000000009,.0000000000000002,-.0000000000000000] 
apsi_cs=cheb_series(apsics_data,15,-1,1,9) 
def psi_x(x,result):
    x_2 = x;result_2 = result;
    c_2=None;c_3=None;c_4=None;result_c_err_IV_0=None;result_c_err_IV_1=None;result_c_err_IV_2=None;result_c_err_IV_3=None;result_c_err_IV_4=None;result_c_err_IV_5=None;result_c_err_IV_6=None;result_c_err_IV_7=None;result_c_err_IV_8=None;result_err_1=None;result_err_2=None;result_err_3=None;result_err_4=None;result_err_5=None;result_err_6=None;result_err_7=None;result_err_8=None;result_err_9=None;result_err_10=None;result_err_11=None;result_err_12=None;result_err_13=None;result_err_14=None;result_err_15=None;result_err_16=None;result_err_17=None;result_err_18=None;result_err_19=None;result_err_20=None;result_err_21=None;result_err_22=None;result_err_23=None;result_err_24=None;result_err_25=None;result_err_26=None;result_err_27=None;result_val_2=None;result_val_3=None;result_val_4=None;result_val_5=None;result_val_6=None;result_val_7=None;result_val_8=None;result_val_9=None;result_val_10=None;s_0=None;s_1=None;s_2=None;t_0=None;t_1=None;result_val_IV_0=None;result_val_IV_1=None;result_val_IV_2=None;result_val_IV_3=None;result_val_IV_4=None;result_val_IV_5=None;result_val_IV_6=None;result_val_IV_7=None;result_val_IV_8=None;v_0=None;v_1=None;v_2=None;v_3=None;v_4=None;y_1=None;result_c_val_IV_0=None;result_c_val_IV_1=None;result_c_val_IV_2=None;result_c_val_IV_3=None;result_c_val_IV_4=None;result_c_val_IV_5=None;result_c_val_IV_6=None;result_c_val_IV_7=None;result_c_val_IV_8=None;result_c_0=None;result_c_1=None;result_c_2=None;t1_0=None;t1_1=None;t1_2=None;t1_3=None;t1_4=None;t2_0=None;t2_1=None;t2_2=None;t2_3=None;t3_0=None;t3_1=None;t3_2=None;

    y_1=fabs(x_2) 
    if x_2==0.0 or x_2==-1.0 or x_2==-2.0:
        print("domain error") 
    elif y_1>=2.0:
        t_0=8.0/(y_1*y_1)-1.0 
        result_c_0=gsl_sf_result(0,0) 
        cheb_eval_e(apsi_cs,t_0,result_c_0) 
        if x_2<0.0:
            s_0=sin(M_PI*x_2) 
            c_2=cos(M_PI*x_2) 
            if fabs(s_0)<2.0*GSL_SQRT_DBL_MIN:
                print("domain error") 
            else:
                result_c_val_IV_0=result_c_0.val 
                result_val_2=log(y_1)-0.5/x_2+result_c_val_IV_0-M_PI*c_2/s_0 
                result_2.val=result_val_2 
                result_err_1=M_PI*fabs(x_2)*GSL_DBL_EPSILON/(s_0*s_0) 
                result_2.err=result_err_1 
                result_err_2=result_2.err 
                result_c_err_IV_0=result_c_0.err 
                result_err_3 = result_err_2+result_c_err_IV_0
                result_2.err=result_err_3 
                result_val_IV_0=result_2.val 
                result_err_4=result_2.err 
                result_err_5 = result_err_4+GSL_DBL_EPSILON*fabs(result_val_IV_0)
                result_2.err=result_err_5 
                lo = locals()
                record_locals(lo, test_counter) 
                return GSL_SUCCESS
            phiPreds = [fabs(s_0)<2.0*GSL_SQRT_DBL_MIN]
            phiNames = [None,result_val_2]
            result_val_3= phiIf(phiPreds, phiNames)
            phiPreds = [fabs(s_0)<2.0*GSL_SQRT_DBL_MIN]
            phiNames = [None,result_val_IV_0]
            result_val_IV_1= phiIf(phiPreds, phiNames)
            phiPreds = [fabs(s_0)<2.0*GSL_SQRT_DBL_MIN]
            phiNames = [None,result_c_err_IV_0]
            result_c_err_IV_1= phiIf(phiPreds, phiNames)
            phiPreds = [fabs(s_0)<2.0*GSL_SQRT_DBL_MIN]
            phiNames = [None,result_c_val_IV_0]
            result_c_val_IV_1= phiIf(phiPreds, phiNames)
            phiPreds = [fabs(s_0)<2.0*GSL_SQRT_DBL_MIN]
            phiNames = [None,result_err_5]
            result_err_6= phiIf(phiPreds, phiNames)
        else:
            result_c_val_IV_2=result_c_0.val 
            result_val_4=log(y_1)-0.5/x_2+result_c_val_IV_2 
            result_2.val=result_val_4 
            result_c_err_IV_2=result_c_0.err 
            result_err_7=result_c_err_IV_2 
            result_2.err=result_err_7 
            result_err_8=result_2.err 
            result_val_IV_2=result_2.val 
            result_err_9 = result_err_8+GSL_DBL_EPSILON*fabs(result_val_IV_2)
            result_2.err=result_err_9 
            lo = locals()
            record_locals(lo, test_counter) 
            return GSL_SUCCESS
        phiPreds = [x_2<0.0]
        phiNames = [result_val_3,result_val_4]
        result_val_5= phiIf(phiPreds, phiNames)
        phiPreds = [x_2<0.0]
        phiNames = [s_0,None]
        s_1= phiIf(phiPreds, phiNames)
        phiPreds = [x_2<0.0]
        phiNames = [c_2,None]
        c_3= phiIf(phiPreds, phiNames)
        phiPreds = [x_2<0.0]
        phiNames = [result_val_IV_1,result_val_IV_2]
        result_val_IV_3= phiIf(phiPreds, phiNames)
        phiPreds = [x_2<0.0]
        phiNames = [result_c_err_IV_1,result_c_err_IV_2]
        result_c_err_IV_3= phiIf(phiPreds, phiNames)
        phiPreds = [x_2<0.0]
        phiNames = [result_c_val_IV_1,result_c_val_IV_2]
        result_c_val_IV_3= phiIf(phiPreds, phiNames)
        phiPreds = [x_2<0.0]
        phiNames = [result_err_6,result_err_9]
        result_err_10= phiIf(phiPreds, phiNames)
    else:
        result_c_1=gsl_sf_result(0,0) 
        if x_2<-1.0:
            v_0=x_2+2.0 
            t1_0=1.0/x_2 
            t2_0=1.0/(x_2+1.0) 
            t3_0=1.0/v_0 
            cheb_eval_e(psi_cs,2.0*v_0-1.0,result_c_1) 
            result_c_val_IV_4=result_c_1.val 
            result_val_6=-(t1_0+t2_0+t3_0)+result_c_val_IV_4 
            result_2.val=result_val_6 
            result_err_11=GSL_DBL_EPSILON*(fabs(t1_0)+fabs(x_2/(t2_0*t2_0))+fabs(x_2/(t3_0*t3_0))) 
            result_2.err=result_err_11 
            result_err_12=result_2.err 
            result_c_err_IV_4=result_c_1.err 
            result_err_13 = result_err_12+result_c_err_IV_4
            result_2.err=result_err_13 
            result_err_14=result_2.err 
            result_val_IV_4=result_2.val 
            result_err_15 = result_err_14+GSL_DBL_EPSILON*fabs(result_val_IV_4)
            result_2.err=result_err_15 
            lo = locals()
            record_locals(lo, test_counter) 
            return GSL_SUCCESS
        elif x_2<0.0:
            v_1=x_2+1.0 
            t1_1=1.0/x_2 
            t2_1=1.0/v_1 
            cheb_eval_e(psi_cs,2.0*v_1-1.0,result_c_1) 
            result_c_val_IV_5=result_c_1.val 
            result_val_7=-(t1_1+t2_1)+result_c_val_IV_5 
            result_2.val=result_val_7 
            result_err_16=GSL_DBL_EPSILON*(fabs(t1_1)+fabs(x_2/(t2_1*t2_1))) 
            result_2.err=result_err_16 
            result_err_17=result_2.err 
            result_c_err_IV_5=result_c_1.err 
            result_err_18 = result_err_17+result_c_err_IV_5
            result_2.err=result_err_18 
            result_err_19=result_2.err 
            result_val_IV_5=result_2.val 
            result_err_20 = result_err_19+GSL_DBL_EPSILON*fabs(result_val_IV_5)
            result_2.err=result_err_20 
            lo = locals()
            record_locals(lo, test_counter) 
            return GSL_SUCCESS
        elif x_2<1.0:
            t1_2=1.0/x_2 
            cheb_eval_e(psi_cs,2.0*x_2-1.0,result_c_1) 
            result_c_val_IV_6=result_c_1.val 
            result_val_8=-t1_2+result_c_val_IV_6 
            result_2.val=result_val_8 
            result_err_21=GSL_DBL_EPSILON*t1_2 
            result_2.err=result_err_21 
            result_err_22=result_2.err 
            result_c_err_IV_6=result_c_1.err 
            result_err_23 = result_err_22+result_c_err_IV_6
            result_2.err=result_err_23 
            result_err_24=result_2.err 
            result_val_IV_6=result_2.val 
            result_err_25 = result_err_24+GSL_DBL_EPSILON*fabs(result_val_IV_6)
            result_2.err=result_err_25
            lo = locals()
            record_locals(lo, test_counter)  
            return GSL_SUCCESS
        else:
            v_2=x_2-1.0 
            lo = locals()
            record_locals(lo, test_counter) 
            return cheb_eval_e(psi_cs,2.0*v_2-1.0,result_2)
        phiPreds = [x_2<-1.0,x_2<0.0,x_2<1.0]
        phiNames = [result_val_6,result_val_7,result_val_8,None]
        result_val_9= phiIf(phiPreds, phiNames)
        phiPreds = [x_2<-1.0,x_2<0.0,x_2<1.0]
        phiNames = [result_val_IV_4,result_val_IV_5,result_val_IV_6,None]
        result_val_IV_7= phiIf(phiPreds, phiNames)
        phiPreds = [x_2<-1.0,x_2<0.0,x_2<1.0]
        phiNames = [v_0,v_1,None,v_2]
        v_3= phiIf(phiPreds, phiNames)
        phiPreds = [x_2<-1.0,x_2<0.0,x_2<1.0]
        phiNames = [result_c_err_IV_4,result_c_err_IV_5,result_c_err_IV_6,None]
        result_c_err_IV_7= phiIf(phiPreds, phiNames)
        phiPreds = [x_2<-1.0,x_2<0.0,x_2<1.0]
        phiNames = [result_c_val_IV_4,result_c_val_IV_5,result_c_val_IV_6,None]
        result_c_val_IV_7= phiIf(phiPreds, phiNames)
        phiPreds = [x_2<-1.0,x_2<0.0,x_2<1.0]
        phiNames = [result_err_15,result_err_20,result_err_25,None]
        result_err_26= phiIf(phiPreds, phiNames)
        phiPreds = [x_2<-1.0,x_2<0.0,x_2<1.0]
        phiNames = [t1_0,t1_1,t1_2,None]
        t1_3= phiIf(phiPreds, phiNames)
        phiPreds = [x_2<-1.0,x_2<0.0,x_2<1.0]
        phiNames = [t2_0,t2_1,None,None]
        t2_2= phiIf(phiPreds, phiNames)
        phiPreds = [x_2<-1.0,x_2<0.0,x_2<1.0]
        phiNames = [t3_0,None,None,None]
        t3_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_2==0.0 or x_2==-1.0 or x_2==-2.0,y_1>=2.0]
    phiNames = [None,c_3,None]
    c_4= phiIf(phiPreds, phiNames)
    phiPreds = [x_2==0.0 or x_2==-1.0 or x_2==-2.0,y_1>=2.0]
    phiNames = [None,result_c_err_IV_3,result_c_err_IV_7]
    result_c_err_IV_8= phiIf(phiPreds, phiNames)
    phiPreds = [x_2==0.0 or x_2==-1.0 or x_2==-2.0,y_1>=2.0]
    phiNames = [None,result_err_10,result_err_26]
    result_err_27= phiIf(phiPreds, phiNames)
    phiPreds = [x_2==0.0 or x_2==-1.0 or x_2==-2.0,y_1>=2.0]
    phiNames = [None,result_val_5,result_val_9]
    result_val_10= phiIf(phiPreds, phiNames)
    phiPreds = [x_2==0.0 or x_2==-1.0 or x_2==-2.0,y_1>=2.0]
    phiNames = [None,s_1,None]
    s_2= phiIf(phiPreds, phiNames)
    phiPreds = [x_2==0.0 or x_2==-1.0 or x_2==-2.0,y_1>=2.0]
    phiNames = [None,t_0,None]
    t_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_2==0.0 or x_2==-1.0 or x_2==-2.0,y_1>=2.0]
    phiNames = [None,result_val_IV_3,result_val_IV_7]
    result_val_IV_8= phiIf(phiPreds, phiNames)
    phiPreds = [x_2==0.0 or x_2==-1.0 or x_2==-2.0,y_1>=2.0]
    phiNames = [None,None,v_3]
    v_4= phiIf(phiPreds, phiNames)
    phiPreds = [x_2==0.0 or x_2==-1.0 or x_2==-2.0,y_1>=2.0]
    phiNames = [None,result_c_val_IV_3,result_c_val_IV_7]
    result_c_val_IV_8= phiIf(phiPreds, phiNames)
    phiPreds = [x_2==0.0 or x_2==-1.0 or x_2==-2.0,y_1>=2.0]
    phiNames = [None,result_c_0,result_c_1]
    result_c_2= phiIf(phiPreds, phiNames)
    phiPreds = [x_2==0.0 or x_2==-1.0 or x_2==-2.0,y_1>=2.0]
    phiNames = [None,None,t1_3]
    t1_4= phiIf(phiPreds, phiNames)
    phiPreds = [x_2==0.0 or x_2==-1.0 or x_2==-2.0,y_1>=2.0]
    phiNames = [None,None,t2_2]
    t2_3= phiIf(phiPreds, phiNames)
    phiPreds = [x_2==0.0 or x_2==-1.0 or x_2==-2.0,y_1>=2.0]
    phiNames = [None,None,t3_1]
    t3_2= phiIf(phiPreds, phiNames)

hzeta_c=[1.00000000000000000000000000000,0.083333333333333333333333333333,-0.00138888888888888888888888888889,0.000033068783068783068783068783069,-8.2671957671957671957671957672e-07,2.0876756987868098979210090321e-08,-5.2841901386874931848476822022e-10,1.3382536530684678832826980975e-11,-3.3896802963225828668301953912e-13,8.5860620562778445641359054504e-15,-2.1748686985580618730415164239e-16,5.5090028283602295152026526089e-18,-1.3954464685812523340707686264e-19,3.5347070396294674716932299778e-21,-8.9535174270375468504026113181e-23] 
def gsl_sf_hzeta_e(s,q,result):
    s_3 = s;q_0 = q;result_3 = result;
    pmax_0=None;pmax_1=None;pmax_2=None;p1_0=None;p1_1=None;p1_2=None;scp_0=None;scp_2=None;scp_1=None;scp_3=None;scp_4=None;scp_5=None;p2_0=None;p2_1=None;p2_2=None;p3_0=None;p3_1=None;p3_2=None;ans_0=None;ans_2=None;ans_1=None;ans_3=None;ans_5=None;ans_4=None;ans_6=None;ans_7=None;ans_8=None;delta_1=None;delta_0=None;delta_2=None;delta_3=None;delta_4=None;max_bits_0=None;max_bits_1=None;jmax_0=None;jmax_1=None;jmax_2=None;kmax_0=None;kmax_1=None;kmax_2=None;result_err_28=None;result_err_29=None;result_err_30=None;result_err_31=None;result_err_32=None;result_val_11=None;result_val_12=None;result_val_13=None;result_val_14=None;result_val_15=None;result_val_IV_9=None;result_val_IV_10=None;result_val_IV_11=None;result_val_IV_12=None;hzeta_c_j_1_IV_1=None;hzeta_c_j_1_IV_0=None;hzeta_c_j_1_IV_2=None;hzeta_c_j_1_IV_3=None;hzeta_c_j_1_IV_4=None;ln_term0_0=None;ln_term0_1=None;pcp_0=None;pcp_2=None;pcp_1=None;pcp_3=None;pcp_4=None;pcp_5=None;

    if s_3<=1.0 or q_0<=0.0:
        print("domain error") 
    else:
        max_bits_0=54.0 
        ln_term0_0=-s_3*log(q_0) 
        if ln_term0_0<GSL_LOG_DBL_MIN+1.0:
            print("underflow error") 
        elif ln_term0_0>GSL_LOG_DBL_MAX-1.0:
            print("overflow error") 
        elif (s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25):
            result_val_11=pow(q_0,-s_3) 
            result_3.val=result_val_11 
            result_val_IV_9=result_3.val 
            result_err_28=2.0*GSL_DBL_EPSILON*fabs(result_val_IV_9) 
            result_3.err=result_err_28 
            lo = locals()
            record_locals(lo, test_counter) 
            return GSL_SUCCESS
        elif s_3>0.5*max_bits_0 and q_0<1.0:
            p1_0=pow(q_0,-s_3) 
            p2_0=pow(q_0/(1.0+q_0),s_3) 
            p3_0=pow(q_0/(2.0+q_0),s_3) 
            result_val_12=p1_0*(1.0+p2_0+p3_0) 
            result_3.val=result_val_12 
            result_val_IV_10=result_3.val 
            result_err_29=GSL_DBL_EPSILON*(0.5*s_3+2.0)*fabs(result_val_IV_10) 
            result_3.err=result_err_29 
            lo = locals()
            record_locals(lo, test_counter) 
            return GSL_SUCCESS
        else:
            jmax_0=12 
            kmax_0=10 
            pmax_0=pow(kmax_0+q_0,-s_3) 
            scp_0=s_3 
            pcp_0=pmax_0/(kmax_0+q_0) 
            ans_0=pmax_0*((kmax_0+q_0)/(s_3-1.0)+0.5) 
            phi0 = Phi()
            for k_0 in range(0,kmax_0):
                phi0.set()
                ans_2 = phi0.phiEntry(ans_0,ans_1)

                ans_1 = ans_2+pow(k_0+q_0,-s_3)
            ans_3 = phi0.phiExit(ans_0,ans_1)
            phi0 = Phi()
            for j_1 in range(0,jmax_0+1):
                phi0.set()
                scp_2 = phi0.phiEntry(scp_0,scp_1)
                ans_5 = phi0.phiEntry(ans_3,ans_4)
                hzeta_c_j_1_IV_1 = phi0.phiEntry(None,hzeta_c_j_1_IV_0)
                delta_1 = phi0.phiEntry(None,delta_0)
                pcp_2 = phi0.phiEntry(pcp_0,pcp_1)

                hzeta_c_j_1_IV_0=hzeta_c[j_1+1] 
                delta_0=hzeta_c_j_1_IV_0*scp_2*pcp_2 
                ans_4 = ans_5+delta_0
                if fabs(delta_0/ans_4)<0.5*GSL_DBL_EPSILON:
                    break
                scp_1 = scp_2*(s_3+2*j_1+1)*(s_3+2*j_1+2)
                pcp_1 = pcp_2/(kmax_0+q_0)*(kmax_0+q_0)
            scp_3 = phi0.phiExit(scp_0,scp_1)
            ans_6 = phi0.phiExit(None,ans_4)
            hzeta_c_j_1_IV_2 = phi0.phiExit(None,hzeta_c_j_1_IV_0)
            delta_2 = phi0.phiExit(None,delta_0)
            pcp_3 = phi0.phiExit(pcp_0,pcp_1)
            result_val_13=ans_6 
            result_3.val=result_val_13 
            result_err_30=2.0*(jmax_0+1.0)*GSL_DBL_EPSILON*fabs(ans_6) 
            result_3.err=result_err_30
            lo = locals()
            record_locals(lo, test_counter)  
            return GSL_SUCCESS
        phiPreds = [ln_term0_0<GSL_LOG_DBL_MIN+1.0,ln_term0_0>GSL_LOG_DBL_MAX-1.0,(s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25),s_3>0.5*max_bits_0 and q_0<1.0]
        phiNames = [None,None,None,None,pmax_0]
        pmax_1= phiIf(phiPreds, phiNames)
        phiPreds = [ln_term0_0<GSL_LOG_DBL_MIN+1.0,ln_term0_0>GSL_LOG_DBL_MAX-1.0,(s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25),s_3>0.5*max_bits_0 and q_0<1.0]
        phiNames = [None,None,None,p1_0,None]
        p1_1= phiIf(phiPreds, phiNames)
        phiPreds = [ln_term0_0<GSL_LOG_DBL_MIN+1.0,ln_term0_0>GSL_LOG_DBL_MAX-1.0,(s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25),s_3>0.5*max_bits_0 and q_0<1.0]
        phiNames = [None,None,None,None,scp_3]
        scp_4= phiIf(phiPreds, phiNames)
        phiPreds = [ln_term0_0<GSL_LOG_DBL_MIN+1.0,ln_term0_0>GSL_LOG_DBL_MAX-1.0,(s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25),s_3>0.5*max_bits_0 and q_0<1.0]
        phiNames = [None,None,None,p2_0,None]
        p2_1= phiIf(phiPreds, phiNames)
        phiPreds = [ln_term0_0<GSL_LOG_DBL_MIN+1.0,ln_term0_0>GSL_LOG_DBL_MAX-1.0,(s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25),s_3>0.5*max_bits_0 and q_0<1.0]
        phiNames = [None,None,None,p3_0,None]
        p3_1= phiIf(phiPreds, phiNames)
        phiPreds = [ln_term0_0<GSL_LOG_DBL_MIN+1.0,ln_term0_0>GSL_LOG_DBL_MAX-1.0,(s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25),s_3>0.5*max_bits_0 and q_0<1.0]
        phiNames = [None,None,None,None,ans_6]
        ans_7= phiIf(phiPreds, phiNames)
        phiPreds = [ln_term0_0<GSL_LOG_DBL_MIN+1.0,ln_term0_0>GSL_LOG_DBL_MAX-1.0,(s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25),s_3>0.5*max_bits_0 and q_0<1.0]
        phiNames = [None,None,None,None,delta_2]
        delta_3= phiIf(phiPreds, phiNames)
        phiPreds = [ln_term0_0<GSL_LOG_DBL_MIN+1.0,ln_term0_0>GSL_LOG_DBL_MAX-1.0,(s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25),s_3>0.5*max_bits_0 and q_0<1.0]
        phiNames = [None,None,None,None,jmax_0]
        jmax_1= phiIf(phiPreds, phiNames)
        phiPreds = [ln_term0_0<GSL_LOG_DBL_MIN+1.0,ln_term0_0>GSL_LOG_DBL_MAX-1.0,(s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25),s_3>0.5*max_bits_0 and q_0<1.0]
        phiNames = [None,None,None,None,kmax_0]
        kmax_1= phiIf(phiPreds, phiNames)
        phiPreds = [ln_term0_0<GSL_LOG_DBL_MIN+1.0,ln_term0_0>GSL_LOG_DBL_MAX-1.0,(s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25),s_3>0.5*max_bits_0 and q_0<1.0]
        phiNames = [None,None,result_err_28,result_err_29,result_err_30]
        result_err_31= phiIf(phiPreds, phiNames)
        phiPreds = [ln_term0_0<GSL_LOG_DBL_MIN+1.0,ln_term0_0>GSL_LOG_DBL_MAX-1.0,(s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25),s_3>0.5*max_bits_0 and q_0<1.0]
        phiNames = [None,None,result_val_11,result_val_12,result_val_13]
        result_val_14= phiIf(phiPreds, phiNames)
        phiPreds = [ln_term0_0<GSL_LOG_DBL_MIN+1.0,ln_term0_0>GSL_LOG_DBL_MAX-1.0,(s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25),s_3>0.5*max_bits_0 and q_0<1.0]
        phiNames = [None,None,result_val_IV_9,result_val_IV_10,None]
        result_val_IV_11= phiIf(phiPreds, phiNames)
        phiPreds = [ln_term0_0<GSL_LOG_DBL_MIN+1.0,ln_term0_0>GSL_LOG_DBL_MAX-1.0,(s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25),s_3>0.5*max_bits_0 and q_0<1.0]
        phiNames = [None,None,None,None,hzeta_c_j_1_IV_2]
        hzeta_c_j_1_IV_3= phiIf(phiPreds, phiNames)
        phiPreds = [ln_term0_0<GSL_LOG_DBL_MIN+1.0,ln_term0_0>GSL_LOG_DBL_MAX-1.0,(s_3>max_bits_0 and q_0<1.0) or (s_3>0.5*max_bits_0 and q_0<0.25),s_3>0.5*max_bits_0 and q_0<1.0]
        phiNames = [None,None,None,None,pcp_3]
        pcp_4= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,pmax_1]
    pmax_2= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,p1_1]
    p1_2= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,scp_4]
    scp_5= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,p2_1]
    p2_2= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,p3_1]
    p3_2= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,ans_7]
    ans_8= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,delta_3]
    delta_4= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,max_bits_0]
    max_bits_1= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,jmax_1]
    jmax_2= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,kmax_1]
    kmax_2= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,result_err_31]
    result_err_32= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,result_val_14]
    result_val_15= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,result_val_IV_11]
    result_val_IV_12= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,hzeta_c_j_1_IV_3]
    hzeta_c_j_1_IV_4= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,ln_term0_0]
    ln_term0_1= phiIf(phiPreds, phiNames)
    phiPreds = [s_3<=1.0 or q_0<=0.0]
    phiNames = [None,pcp_4]
    pcp_5= phiIf(phiPreds, phiNames)

def gsl_sf_exp_mult_err_e(x,dx,y,dy,result):
    x_3 = x;dx_0 = dx;y_2 = y;dy_0 = dy;result_4 = result;
    a_3=None;a_4=None;a_5=None;b_3=None;b_4=None;b_5=None;sy_0=None;sy_1=None;sy_2=None;eMN_0=None;eMN_1=None;eMN_2=None;eab_0=None;eab_1=None;eab_2=None;result_err_33=None;result_err_34=None;result_err_35=None;result_err_36=None;result_err_37=None;result_err_38=None;result_err_39=None;result_err_40=None;result_err_41=None;result_err_42=None;result_err_43=None;ly_0=None;ly_1=None;M_0=None;M_1=None;M_2=None;N_0=None;N_1=None;N_2=None;result_val_16=None;result_val_17=None;result_val_18=None;result_val_19=None;result_val_20=None;lnr_0=None;lnr_1=None;ex_0=None;ex_1=None;result_val_IV_13=None;result_val_IV_14=None;ay_0=None;

    ay_0=fabs(y_2) 
    if y_2==0.0:
        result_val_16=0.0 
        result_4.val=result_val_16 
        result_err_33=fabs(dy_0*exp(x_3)) 
        result_4.err=result_err_33 
        lo = locals()
        record_locals(lo, test_counter) 
        return GSL_SUCCESS
    elif (x_3<0.5*GSL_LOG_DBL_MAX and x_3>0.5*GSL_LOG_DBL_MIN) and (ay_0<0.8*GSL_SQRT_DBL_MAX and ay_0>1.2*GSL_SQRT_DBL_MIN):
        ex_0=exp(x_3) 
        result_val_17=y_2*ex_0 
        result_4.val=result_val_17 
        result_err_34=ex_0*(fabs(dy_0)+fabs(y_2*dx_0)) 
        result_4.err=result_err_34 
        result_err_35=result_4.err 
        result_val_IV_13=result_4.val 
        result_err_36 = result_err_35+2.0*GSL_DBL_EPSILON*fabs(result_val_IV_13)
        result_4.err=result_err_36 
        lo = locals()
        record_locals(lo, test_counter) 
        return GSL_SUCCESS
    else:
        ly_0=log(ay_0) 
        lnr_0=x_3+ly_0 
        if lnr_0>GSL_LOG_DBL_MAX-0.01:
            print("overflow error") 
        elif lnr_0<GSL_LOG_DBL_MIN+0.01:
            print("underflow error") 
        else:
            sy_0=GSL_SIGN(y_2) 
            M_0=floor(x_3) 
            N_0=floor(ly_0) 
            a_3=x_3-M_0 
            b_3=ly_0-N_0 
            eMN_0=exp(M_0+N_0) 
            eab_0=exp(a_3+b_3) 
            result_val_18=sy_0*eMN_0*eab_0 
            result_4.val=result_val_18 
            result_err_37=eMN_0*eab_0*2.0*GSL_DBL_EPSILON 
            result_4.err=result_err_37 
            result_err_38=result_4.err 
            result_err_39 = result_err_38+eMN_0*eab_0*fabs(dy_0/y_2)
            result_4.err=result_err_39 
            result_err_40=result_4.err 
            result_err_41 = result_err_40+eMN_0*eab_0*fabs(dx_0)
            result_4.err=result_err_41
            lo = locals()
            record_locals(lo, test_counter)  
            return GSL_SUCCESS
        phiPreds = [lnr_0>GSL_LOG_DBL_MAX-0.01,lnr_0<GSL_LOG_DBL_MIN+0.01]
        phiNames = [None,None,result_val_18]
        result_val_19= phiIf(phiPreds, phiNames)
        phiPreds = [lnr_0>GSL_LOG_DBL_MAX-0.01,lnr_0<GSL_LOG_DBL_MIN+0.01]
        phiNames = [None,None,a_3]
        a_4= phiIf(phiPreds, phiNames)
        phiPreds = [lnr_0>GSL_LOG_DBL_MAX-0.01,lnr_0<GSL_LOG_DBL_MIN+0.01]
        phiNames = [None,None,b_3]
        b_4= phiIf(phiPreds, phiNames)
        phiPreds = [lnr_0>GSL_LOG_DBL_MAX-0.01,lnr_0<GSL_LOG_DBL_MIN+0.01]
        phiNames = [None,None,sy_0]
        sy_1= phiIf(phiPreds, phiNames)
        phiPreds = [lnr_0>GSL_LOG_DBL_MAX-0.01,lnr_0<GSL_LOG_DBL_MIN+0.01]
        phiNames = [None,None,eMN_0]
        eMN_1= phiIf(phiPreds, phiNames)
        phiPreds = [lnr_0>GSL_LOG_DBL_MAX-0.01,lnr_0<GSL_LOG_DBL_MIN+0.01]
        phiNames = [None,None,eab_0]
        eab_1= phiIf(phiPreds, phiNames)
        phiPreds = [lnr_0>GSL_LOG_DBL_MAX-0.01,lnr_0<GSL_LOG_DBL_MIN+0.01]
        phiNames = [None,None,result_err_41]
        result_err_42= phiIf(phiPreds, phiNames)
        phiPreds = [lnr_0>GSL_LOG_DBL_MAX-0.01,lnr_0<GSL_LOG_DBL_MIN+0.01]
        phiNames = [None,None,M_0]
        M_1= phiIf(phiPreds, phiNames)
        phiPreds = [lnr_0>GSL_LOG_DBL_MAX-0.01,lnr_0<GSL_LOG_DBL_MIN+0.01]
        phiNames = [None,None,N_0]
        N_1= phiIf(phiPreds, phiNames)
    phiPreds = [y_2==0.0,(x_3<0.5*GSL_LOG_DBL_MAX and x_3>0.5*GSL_LOG_DBL_MIN) and (ay_0<0.8*GSL_SQRT_DBL_MAX and ay_0>1.2*GSL_SQRT_DBL_MIN)]
    phiNames = [None,None,a_4]
    a_5= phiIf(phiPreds, phiNames)
    phiPreds = [y_2==0.0,(x_3<0.5*GSL_LOG_DBL_MAX and x_3>0.5*GSL_LOG_DBL_MIN) and (ay_0<0.8*GSL_SQRT_DBL_MAX and ay_0>1.2*GSL_SQRT_DBL_MIN)]
    phiNames = [None,None,b_4]
    b_5= phiIf(phiPreds, phiNames)
    phiPreds = [y_2==0.0,(x_3<0.5*GSL_LOG_DBL_MAX and x_3>0.5*GSL_LOG_DBL_MIN) and (ay_0<0.8*GSL_SQRT_DBL_MAX and ay_0>1.2*GSL_SQRT_DBL_MIN)]
    phiNames = [None,None,sy_1]
    sy_2= phiIf(phiPreds, phiNames)
    phiPreds = [y_2==0.0,(x_3<0.5*GSL_LOG_DBL_MAX and x_3>0.5*GSL_LOG_DBL_MIN) and (ay_0<0.8*GSL_SQRT_DBL_MAX and ay_0>1.2*GSL_SQRT_DBL_MIN)]
    phiNames = [None,None,eMN_1]
    eMN_2= phiIf(phiPreds, phiNames)
    phiPreds = [y_2==0.0,(x_3<0.5*GSL_LOG_DBL_MAX and x_3>0.5*GSL_LOG_DBL_MIN) and (ay_0<0.8*GSL_SQRT_DBL_MAX and ay_0>1.2*GSL_SQRT_DBL_MIN)]
    phiNames = [None,None,eab_1]
    eab_2= phiIf(phiPreds, phiNames)
    phiPreds = [y_2==0.0,(x_3<0.5*GSL_LOG_DBL_MAX and x_3>0.5*GSL_LOG_DBL_MIN) and (ay_0<0.8*GSL_SQRT_DBL_MAX and ay_0>1.2*GSL_SQRT_DBL_MIN)]
    phiNames = [result_err_33,result_err_36,result_err_42]
    result_err_43= phiIf(phiPreds, phiNames)
    phiPreds = [y_2==0.0,(x_3<0.5*GSL_LOG_DBL_MAX and x_3>0.5*GSL_LOG_DBL_MIN) and (ay_0<0.8*GSL_SQRT_DBL_MAX and ay_0>1.2*GSL_SQRT_DBL_MIN)]
    phiNames = [None,None,ly_0]
    ly_1= phiIf(phiPreds, phiNames)
    phiPreds = [y_2==0.0,(x_3<0.5*GSL_LOG_DBL_MAX and x_3>0.5*GSL_LOG_DBL_MIN) and (ay_0<0.8*GSL_SQRT_DBL_MAX and ay_0>1.2*GSL_SQRT_DBL_MIN)]
    phiNames = [None,None,M_1]
    M_2= phiIf(phiPreds, phiNames)
    phiPreds = [y_2==0.0,(x_3<0.5*GSL_LOG_DBL_MAX and x_3>0.5*GSL_LOG_DBL_MIN) and (ay_0<0.8*GSL_SQRT_DBL_MAX and ay_0>1.2*GSL_SQRT_DBL_MIN)]
    phiNames = [None,None,N_1]
    N_2= phiIf(phiPreds, phiNames)
    phiPreds = [y_2==0.0,(x_3<0.5*GSL_LOG_DBL_MAX and x_3>0.5*GSL_LOG_DBL_MIN) and (ay_0<0.8*GSL_SQRT_DBL_MAX and ay_0>1.2*GSL_SQRT_DBL_MIN)]
    phiNames = [result_val_16,result_val_17,result_val_19]
    result_val_20= phiIf(phiPreds, phiNames)
    phiPreds = [y_2==0.0,(x_3<0.5*GSL_LOG_DBL_MAX and x_3>0.5*GSL_LOG_DBL_MIN) and (ay_0<0.8*GSL_SQRT_DBL_MAX and ay_0>1.2*GSL_SQRT_DBL_MIN)]
    phiNames = [None,None,lnr_0]
    lnr_1= phiIf(phiPreds, phiNames)
    phiPreds = [y_2==0.0,(x_3<0.5*GSL_LOG_DBL_MAX and x_3>0.5*GSL_LOG_DBL_MIN) and (ay_0<0.8*GSL_SQRT_DBL_MAX and ay_0>1.2*GSL_SQRT_DBL_MIN)]
    phiNames = [None,ex_0,None]
    ex_1= phiIf(phiPreds, phiNames)
    phiPreds = [y_2==0.0,(x_3<0.5*GSL_LOG_DBL_MAX and x_3>0.5*GSL_LOG_DBL_MIN) and (ay_0<0.8*GSL_SQRT_DBL_MAX and ay_0>1.2*GSL_SQRT_DBL_MIN)]
    phiNames = [None,result_val_IV_13,None]
    result_val_IV_14= phiIf(phiPreds, phiNames)

def gsl_sf_psi_e(x,result):
    x_4 = x;result_5 = result;
    

    return psi_x(x_4,result_5)

def psi_n_xg0(n,x,result):
    n_2 = n;x_5 = x;result_6 = result;
    hzeta_val_IV_0=None;hzeta_val_IV_1=None;stat_nf_0=None;stat_nf_1=None;ln_nf_err_IV_0=None;ln_nf_err_IV_1=None;ln_nf_0=None;ln_nf_1=None;result_val_21=None;result_val_22=None;result_val_23=None;hzeta_0=None;hzeta_1=None;result_val_IV_15=None;result_val_IV_16=None;result_val_IV_17=None;stat_hz_0=None;stat_hz_1=None;stat_e_0=None;stat_e_1=None;ln_nf_val_IV_0=None;ln_nf_val_IV_1=None;hzeta_err_IV_0=None;hzeta_err_IV_1=None;

    if n_2==0:
        lo = locals()
        record_locals(lo, test_counter) 
        return gsl_sf_psi_e(x_5,result_6)
    else:
        ln_nf_0=gsl_sf_result(0,0) 
        hzeta_0=gsl_sf_result(0,0) 
        stat_hz_0=gsl_sf_hzeta_e(n_2+1.0,x_5,hzeta_0) 
        stat_nf_0=gsl_sf_lnfact_e(n_2,ln_nf_0) 
        ln_nf_val_IV_0=ln_nf_0.val 
        ln_nf_err_IV_0=ln_nf_0.err 
        hzeta_val_IV_0=hzeta_0.val 
        hzeta_err_IV_0=hzeta_0.err 
        stat_e_0=gsl_sf_exp_mult_err_e(ln_nf_val_IV_0,ln_nf_err_IV_0,hzeta_val_IV_0,hzeta_err_IV_0,result_6) 
        if GSL_IS_EVEN(n_2):
            result_val_IV_15=result_6.val 
            result_val_21=-result_val_IV_15 
            result_6.val=result_val_21 
        phiPreds = [GSL_IS_EVEN(n_2)]
        phiNames = [result_val_21,None]
        result_val_22= phiIf(phiPreds, phiNames)
        phiPreds = [GSL_IS_EVEN(n_2)]
        phiNames = [result_val_IV_15,None]
        result_val_IV_16= phiIf(phiPreds, phiNames)
        lo = locals()
        record_locals(lo, test_counter) 
        return GSL_ERROR_SELECT_3(stat_e_0,stat_nf_0,stat_hz_0)
    phiPreds = [n_2==0]
    phiNames = [None,ln_nf_0]
    ln_nf_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_2==0]
    phiNames = [None,result_val_22]
    result_val_23= phiIf(phiPreds, phiNames)
    phiPreds = [n_2==0]
    phiNames = [None,hzeta_val_IV_0]
    hzeta_val_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_2==0]
    phiNames = [None,hzeta_0]
    hzeta_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_2==0]
    phiNames = [None,result_val_IV_16]
    result_val_IV_17= phiIf(phiPreds, phiNames)
    phiPreds = [n_2==0]
    phiNames = [None,stat_hz_0]
    stat_hz_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_2==0]
    phiNames = [None,stat_e_0]
    stat_e_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_2==0]
    phiNames = [None,stat_nf_0]
    stat_nf_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_2==0]
    phiNames = [None,ln_nf_val_IV_0]
    ln_nf_val_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_2==0]
    phiNames = [None,ln_nf_err_IV_0]
    ln_nf_err_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_2==0]
    phiNames = [None,hzeta_err_IV_0]
    hzeta_err_IV_1= phiIf(phiPreds, phiNames)

def gsl_sf_psi_1_e(x,result):
    x_6 = x;result_7 = result;
    r_val_IV_0=None;r_val_IV_1=None;d_5=None;d_6=None;sum_0=None;sum_2=None;sum_1=None;sum_3=None;sum_4=None;result_err_44=None;result_err_45=None;result_err_46=None;result_err_47=None;M_3=None;M_4=None;result_val_24=None;result_val_25=None;result_val_26=None;result_val_27=None;fx_0=None;fx_1=None;r_0=None;r_1=None;stat_psi_0=None;stat_psi_1=None;stat_psi_2=None;sin_px_0=None;sin_px_1=None;r_err_IV_0=None;r_err_IV_1=None;

    if x_6==0.0 or x_6==-1.0 or x_6==-2.0:
        print("domain error") 
    elif x_6>0.0:
        lo = locals()
        record_locals(lo, test_counter) 
        return psi_n_xg0(1,x_6,result_7)
    elif x_6>-5.0:
        M_3=-floor(x_6) 
        fx_0=x_6+M_3 
        sum_0=0.0 
        if fx_0==0.0:
            print("domain error") 
        phi0 = Phi()
        for m_0 in range(0,M_3):
            phi0.set()
            sum_2 = phi0.phiEntry(sum_0,sum_1)

            sum_1 = sum_2+1.0/((x_6+m_0)*(x_6+m_0))
        sum_3 = phi0.phiExit(sum_0,sum_1)
        stat_psi_0=psi_n_xg0(1,fx_0,result_7) 
        result_val_24=result_7.val 
        result_val_25 = result_val_24+sum_3
        result_7.val=result_val_25 
        result_err_44=result_7.err 
        result_err_45 = result_err_44+M_3*GSL_DBL_EPSILON*sum_3
        result_7.err=result_err_45 
        lo = locals()
        record_locals(lo, test_counter) 
        return stat_psi_0
    else:
        sin_px_0=sin(M_PI*x_6) 
        d_5=M_PI*M_PI/(sin_px_0*sin_px_0) 
        r_0=gsl_sf_result(0,0) 
        stat_psi_1=psi_n_xg0(1,1.0-x_6,r_0) 
        r_val_IV_0=r_0.val 
        result_val_26=d_5-r_val_IV_0 
        result_7.val=result_val_26 
        r_err_IV_0=r_0.err 
        result_err_46=r_err_IV_0+2.0*GSL_DBL_EPSILON*d_5 
        result_7.err=result_err_46 
        lo = locals()
        record_locals(lo, test_counter) 
        return stat_psi_1
    phiPreds = [x_6==0.0 or x_6==-1.0 or x_6==-2.0,x_6>0.0,x_6>-5.0]
    phiNames = [None,None,result_val_25,result_val_26]
    result_val_27= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0 or x_6==-1.0 or x_6==-2.0,x_6>0.0,x_6>-5.0]
    phiNames = [None,None,fx_0,None]
    fx_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0 or x_6==-1.0 or x_6==-2.0,x_6>0.0,x_6>-5.0]
    phiNames = [None,None,None,r_0]
    r_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0 or x_6==-1.0 or x_6==-2.0,x_6>0.0,x_6>-5.0]
    phiNames = [None,None,None,r_val_IV_0]
    r_val_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0 or x_6==-1.0 or x_6==-2.0,x_6>0.0,x_6>-5.0]
    phiNames = [None,None,stat_psi_0,stat_psi_1]
    stat_psi_2= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0 or x_6==-1.0 or x_6==-2.0,x_6>0.0,x_6>-5.0]
    phiNames = [None,None,None,d_5]
    d_6= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0 or x_6==-1.0 or x_6==-2.0,x_6>0.0,x_6>-5.0]
    phiNames = [None,None,None,sin_px_0]
    sin_px_1= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0 or x_6==-1.0 or x_6==-2.0,x_6>0.0,x_6>-5.0]
    phiNames = [None,None,sum_3,None]
    sum_4= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0 or x_6==-1.0 or x_6==-2.0,x_6>0.0,x_6>-5.0]
    phiNames = [None,None,result_err_45,result_err_46]
    result_err_47= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0 or x_6==-1.0 or x_6==-2.0,x_6>0.0,x_6>-5.0]
    phiNames = [None,None,M_3,None]
    M_4= phiIf(phiPreds, phiNames)
    phiPreds = [x_6==0.0 or x_6==-1.0 or x_6==-2.0,x_6>0.0,x_6>-5.0]
    phiNames = [None,None,None,r_err_IV_0]
    r_err_IV_1= phiIf(phiPreds, phiNames)

def gsl_sf_psi_n_e(n,x,result):
    n_3 = n;x_7 = x;result_8 = result;
    hzeta_val_IV_2=None;hzeta_val_IV_3=None;stat_nf_2=None;stat_nf_3=None;ln_nf_err_IV_2=None;ln_nf_err_IV_3=None;ln_nf_2=None;ln_nf_3=None;result_val_28=None;result_val_29=None;result_val_30=None;hzeta_2=None;hzeta_3=None;result_val_IV_18=None;result_val_IV_19=None;result_val_IV_20=None;stat_hz_2=None;stat_hz_3=None;stat_e_2=None;stat_e_3=None;ln_nf_val_IV_2=None;ln_nf_val_IV_3=None;hzeta_err_IV_2=None;hzeta_err_IV_3=None;

    if n_3==0:
        lo = locals()
        record_locals(lo, test_counter) 
        return gsl_sf_psi_e(x_7,result_8)
    elif n_3==1:
        lo = locals()
        record_locals(lo, test_counter) 
        return gsl_sf_psi_1_e(x_7,result_8)
    elif n_3<0 or x_7<=0.0:
        print("domain error") 
    else:
        ln_nf_2=gsl_sf_result(0,0) 
        hzeta_2=gsl_sf_result(0,0) 
        stat_hz_2=gsl_sf_hzeta_e(n_3+1.0,x_7,hzeta_2) 
        stat_nf_2=gsl_sf_lnfact_e(n_3,ln_nf_2) 
        ln_nf_val_IV_2=ln_nf_2.val 
        ln_nf_err_IV_2=ln_nf_2.err 
        hzeta_val_IV_2=hzeta_2.val 
        hzeta_err_IV_2=hzeta_2.err 
        stat_e_2=gsl_sf_exp_mult_err_e(ln_nf_val_IV_2,ln_nf_err_IV_2,hzeta_val_IV_2,hzeta_err_IV_2,result_8) 
        if GSL_IS_EVEN(n_3):
            result_val_IV_18=result_8.val 
            result_val_28=-result_val_IV_18 
            result_8.val=result_val_28 
        phiPreds = [GSL_IS_EVEN(n_3)]
        phiNames = [result_val_28,None]
        result_val_29= phiIf(phiPreds, phiNames)
        phiPreds = [GSL_IS_EVEN(n_3)]
        phiNames = [result_val_IV_18,None]
        result_val_IV_19= phiIf(phiPreds, phiNames)
        lo = locals()
        record_locals(lo, test_counter) 
        return GSL_ERROR_SELECT_3(stat_e_2,stat_nf_2,stat_hz_2)
    phiPreds = [n_3==0,n_3==1,n_3<0 or x_7<=0.0]
    phiNames = [None,None,None,ln_nf_2]
    ln_nf_3= phiIf(phiPreds, phiNames)
    phiPreds = [n_3==0,n_3==1,n_3<0 or x_7<=0.0]
    phiNames = [None,None,None,result_val_29]
    result_val_30= phiIf(phiPreds, phiNames)
    phiPreds = [n_3==0,n_3==1,n_3<0 or x_7<=0.0]
    phiNames = [None,None,None,hzeta_val_IV_2]
    hzeta_val_IV_3= phiIf(phiPreds, phiNames)
    phiPreds = [n_3==0,n_3==1,n_3<0 or x_7<=0.0]
    phiNames = [None,None,None,hzeta_2]
    hzeta_3= phiIf(phiPreds, phiNames)
    phiPreds = [n_3==0,n_3==1,n_3<0 or x_7<=0.0]
    phiNames = [None,None,None,result_val_IV_19]
    result_val_IV_20= phiIf(phiPreds, phiNames)
    phiPreds = [n_3==0,n_3==1,n_3<0 or x_7<=0.0]
    phiNames = [None,None,None,stat_hz_2]
    stat_hz_3= phiIf(phiPreds, phiNames)
    phiPreds = [n_3==0,n_3==1,n_3<0 or x_7<=0.0]
    phiNames = [None,None,None,stat_e_2]
    stat_e_3= phiIf(phiPreds, phiNames)
    phiPreds = [n_3==0,n_3==1,n_3<0 or x_7<=0.0]
    phiNames = [None,None,None,stat_nf_2]
    stat_nf_3= phiIf(phiPreds, phiNames)
    phiPreds = [n_3==0,n_3==1,n_3<0 or x_7<=0.0]
    phiNames = [None,None,None,ln_nf_val_IV_2]
    ln_nf_val_IV_3= phiIf(phiPreds, phiNames)
    phiPreds = [n_3==0,n_3==1,n_3<0 or x_7<=0.0]
    phiNames = [None,None,None,ln_nf_err_IV_2]
    ln_nf_err_IV_3= phiIf(phiPreds, phiNames)
    phiPreds = [n_3==0,n_3==1,n_3<0 or x_7<=0.0]
    phiNames = [None,None,None,hzeta_err_IV_2]
    hzeta_err_IV_3= phiIf(phiPreds, phiNames)

PSI_1_TABLE_NMAX=100 
psi_1_table=[0.0,M_PI*M_PI/6.0,0.644934066848226436472415,0.394934066848226436472415,0.2838229557371153253613041,0.2213229557371153253613041,0.1813229557371153253613041,0.1535451779593375475835263,0.1331370146940314251345467,0.1175120146940314251345467,0.1051663356816857461222010,0.0951663356816857461222010,0.0869018728717683907503002,0.0799574284273239463058557,0.0740402686640103368384001,0.0689382278476838062261552,0.0644937834032393617817108,0.0605875334032393617817108,0.0571273257907826143768665,0.0540409060376961946237801,0.0512708229352031198315363,0.0487708229352031198315363,0.0465032492390579951149830,0.0444371335365786562720078,0.0425467743683366902984728,0.0408106632572255791873617,0.0392106632572255791873617,0.0377313733163971768204978,0.0363596312039143235969038,0.0350841209998326909438426,0.0338950603577399442137594,0.0327839492466288331026483,0.0317433665203020901265817,0.03076680402030209012658168,0.02984853037475571730748159,0.02898347847164153045627052,0.02816715194102928555831133,0.02739554700275768062003973,0.02666508681283803124093089,0.02597256603721476254286995,0.02531510384129102815759710,0.02469010384129102815759710,0.02409521984367056414807896,0.02352832641963428296894063,0.02298749353699501850166102,0.02247096461137518379091722,0.02197713745088135663042339,0.02150454765882086513703965,0.02105185413233829383780923,0.02061782635456051606003145,0.02020133322669712580597065,0.01980133322669712580597065,0.01941686571420193164987683,0.01904704322899483105816086,0.01869104465298913508094477,0.01834810912486842177504628,0.01801753061247172756017024,0.01769865306145131939690494,0.01739086605006319997554452,0.01709360088954001329302371,0.01680632711763538818529605,0.01652854933985761040751827,0.01625980437882562975715546,0.01599965869724394401313881,0.01574770606433893015574400,0.01550356543933893015574400,0.01526687904880638577704578,0.01503731063741979257227076,0.01481454387422086185273411,0.01459828089844231513993134,0.01438824099085987447620523,0.01418415935820681325171544,0.01398578601958352422176106,0.01379288478501562298719316,0.01360523231738567365335942,0.01342261726990576130858221,0.01324483949212798353080444,0.01307170929822216635628920,0.01290304679189732236910755,0.01273868124291638877278934,0.01257845051066194236996928,0.01242220051066194236996928,0.01226978472038606978956995,0.01212106372098095378719041,0.01197590477193174490346273,0.01183418141592267460867815,0.01169577311142440471248438,0.01156056489076458859566448,0.01142844704164317229232189,0.01129931481023821361463594,0.01117306812421372175754719,0.01104961133409026496742374,0.01092885297157366069257770,0.01081070552355853781923177,0.01069508522063334415522437,0.01058191183901270133041676,0.01047110851491297833872701,0.01036260157046853389428257,0.01025632035036012704977199,0.01015219706839427948625679,0.01005016666333357139524567] 
PSI_TABLE_NMAX=100 
psi_table=[0.0,-M_EULER,0.42278433509846713939348790992,0.92278433509846713939348790992,1.25611766843180047272682124325,1.50611766843180047272682124325,1.70611766843180047272682124325,1.87278433509846713939348790992,2.01564147795560999653634505277,2.14064147795560999653634505277,2.25175258906672110764745616389,2.35175258906672110764745616389,2.44266167997581201673836525479,2.52599501330914535007169858813,2.60291809023222227314862166505,2.67434666166079370172005023648,2.74101332832746036838671690315,2.80351332832746036838671690315,2.86233685773922507426906984432,2.91789241329478062982462539988,2.97052399224214905087725697883,3.02052399224214905087725697883,3.06814303986119666992487602645,3.11359758531574212447033057190,3.15707584618530734186163491973,3.1987425128519740085283015864,3.2387425128519740085283015864,3.2772040513135124700667631249,3.3142410883505495071038001619,3.3499553740648352213895144476,3.3844381326855248765619282407,3.4177714660188582098952615740,3.4500295305349872421533260902,3.4812795305349872421533260902,3.5115825608380175451836291205,3.5409943255438998981248055911,3.5695657541153284695533770196,3.5973435318931062473311547974,3.6243705589201332743581818244,3.6506863483938174848844976139,3.6763273740348431259101386396,3.7013273740348431259101386396,3.7257176179372821503003825420,3.7495271417468059598241920658,3.7727829557002943319172153216,3.7955102284275670591899425943,3.8177324506497892814121648166,3.8394715810845718901078169905,3.8607481768292527411716467777,3.8815815101625860745049801110,3.9019896734278921969539597029,3.9219896734278921969539597029,3.9415975165651470989147440166,3.9608282857959163296839747858,3.9796962103242182164764276160,3.9982147288427367349949461345,4.0163965470245549168131279527,4.0342536898816977739559850956,4.0517975495308205809735289552,4.0690389288411654085597358518,4.0859880813835382899156680552,4.1026547480502049565823347218,4.1190481906731557762544658694,4.1351772229312202923834981274,4.1510502388042361653993711433,4.1666752388042361653993711433,4.1820598541888515500147557587,4.1972113693403667015299072739,4.2121367424746950597388624977,4.2268426248276362362094507330,4.2413353784508246420065521823,4.2556210927365389277208378966,4.2697055997787924488475984600,4.2835944886676813377364873489,4.2972931188046676391063503626,4.3108066323181811526198638761,4.3241399656515144859531972094,4.3372978603883565912163551041,4.3502848733753695782293421171,4.3631053861958823987421626300,4.3757636140439836645649474401,4.3882636140439836645649474401,4.4006092930563293435772931191,4.4128044150075488557724150703,4.4248526077786331931218126607,4.4367573696833950978837174226,4.4485220755657480390601880108,4.4601499825424922251066996387,4.4716442354160554434975042364,4.4830078717796918071338678728,4.4942438268358715824147667492,4.5053549379469826935258778603,4.5163439489359936825368668713,4.5272135141533849868846929582,4.5379662023254279976373811303,4.5486045001977684231692960239,4.5591308159872421073798223397,4.5695474826539087740464890064,4.5798567610044242379640147796,4.5900608426370772991885045755,4.6001618527380874001986055856] 
def gsl_sf_psi_1_int_e(n,result):
    n_4 = n;result_9 = result;
    result_val_31=None;result_val_32=None;result_val_33=None;ser_0=None;ser_1=None;result_val_IV_21=None;result_val_IV_22=None;result_val_IV_23=None;ni2_0=None;ni2_1=None;result_err_48=None;result_err_49=None;result_err_50=None;c0_0=None;c0_1=None;c1_0=None;c1_1=None;psi_1_table_n_IV_0=None;psi_1_table_n_IV_1=None;c2_0=None;c2_1=None;

    if n_4<=0:
        print('domain error') 
    elif n_4<=PSI_1_TABLE_NMAX:
        psi_1_table_n_IV_0=psi_1_table[n_4] 
        result_val_31=psi_1_table_n_IV_0 
        result_9.val=result_val_31 
        result_val_IV_21=result_9.val 
        result_err_48=GSL_DBL_EPSILON*result_val_IV_21 
        result_9.err=result_err_48
        lo = locals()
        record_locals(lo, test_counter)  
        return GSL_SUCCESS
    else:
        c0_0=-1.0/30.0 
        c1_0=1.0/42.0 
        c2_0=-1.0/30.0 
        ni2_0=(1.0/n_4)*(1.0/n_4) 
        ser_0=ni2_0*ni2_0*(c0_0+ni2_0*(c1_0+c2_0*ni2_0)) 
        result_val_32=(1.0+0.5/n_4+1.0/(6.0*n_4*n_4)+ser_0)/n_4 
        result_9.val=result_val_32 
        result_val_IV_22=result_9.val 
        result_err_49=GSL_DBL_EPSILON*result_val_IV_22 
        result_9.err=result_err_49
        lo = locals()
        record_locals(lo, test_counter)  
        return GSL_SUCCESS
    phiPreds = [n_4<=0,n_4<=PSI_1_TABLE_NMAX]
    phiNames = [None,result_val_31,result_val_32]
    result_val_33= phiIf(phiPreds, phiNames)
    phiPreds = [n_4<=0,n_4<=PSI_1_TABLE_NMAX]
    phiNames = [None,None,ser_0]
    ser_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_4<=0,n_4<=PSI_1_TABLE_NMAX]
    phiNames = [None,result_val_IV_21,result_val_IV_22]
    result_val_IV_23= phiIf(phiPreds, phiNames)
    phiPreds = [n_4<=0,n_4<=PSI_1_TABLE_NMAX]
    phiNames = [None,None,ni2_0]
    ni2_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_4<=0,n_4<=PSI_1_TABLE_NMAX]
    phiNames = [None,result_err_48,result_err_49]
    result_err_50= phiIf(phiPreds, phiNames)
    phiPreds = [n_4<=0,n_4<=PSI_1_TABLE_NMAX]
    phiNames = [None,None,c0_0]
    c0_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_4<=0,n_4<=PSI_1_TABLE_NMAX]
    phiNames = [None,None,c1_0]
    c1_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_4<=0,n_4<=PSI_1_TABLE_NMAX]
    phiNames = [None,psi_1_table_n_IV_0,None]
    psi_1_table_n_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_4<=0,n_4<=PSI_1_TABLE_NMAX]
    phiNames = [None,None,c2_0]
    c2_1= phiIf(phiPreds, phiNames)

def gsl_sf_psi_int_e(n,result):
    n_5 = n;result_10 = result;
    result_val_34=None;result_val_35=None;result_val_36=None;c3_0=None;c3_1=None;c4_0=None;c4_1=None;ser_2=None;ser_3=None;c5_0=None;c5_1=None;result_val_IV_24=None;result_val_IV_25=None;result_val_IV_26=None;ni2_2=None;ni2_3=None;psi_table_n_IV_0=None;psi_table_n_IV_1=None;result_err_51=None;result_err_52=None;result_err_53=None;result_err_54=None;result_err_55=None;c2_2=None;c2_3=None;

    if n_5<=0:
        print('domain error') 
    elif n_5<=PSI_TABLE_NMAX:
        psi_table_n_IV_0=psi_table[n_5] 
        result_val_34=psi_table_n_IV_0 
        result_10.val=result_val_34 
        result_val_IV_24=result_10.val 
        result_err_51=GSL_DBL_EPSILON*fabs(result_val_IV_24) 
        result_10.err=result_err_51
        lo = locals()
        record_locals(lo, test_counter)  
        return GSL_SUCCESS
    else:
        c2_2=-1.0/12.0 
        c3_0=1.0/120.0 
        c4_0=-1.0/252.0 
        c5_0=1.0/240.0 
        ni2_2=(1.0/n_5)*(1.0/n_5) 
        ser_2=ni2_2*(c2_2+ni2_2*(c3_0+ni2_2*(c4_0+ni2_2*c5_0))) 
        result_val_35=log(n_5)-0.5/n_5+ser_2 
        result_10.val=result_val_35 
        result_err_52=GSL_DBL_EPSILON*(fabs(log(n_5))+fabs(0.5/n_5)+fabs(ser_2)) 
        result_10.err=result_err_52 
        result_val_IV_25=result_10.val 
        result_err_53=result_10.err 
        result_err_54 = result_err_53+GSL_DBL_EPSILON*fabs(result_val_IV_25)
        result_10.err=result_err_54
        lo = locals()
        record_locals(lo, test_counter)  
        return GSL_SUCCESS
    phiPreds = [n_5<=0,n_5<=PSI_TABLE_NMAX]
    phiNames = [None,result_val_34,result_val_35]
    result_val_36= phiIf(phiPreds, phiNames)
    phiPreds = [n_5<=0,n_5<=PSI_TABLE_NMAX]
    phiNames = [None,None,c3_0]
    c3_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_5<=0,n_5<=PSI_TABLE_NMAX]
    phiNames = [None,None,c4_0]
    c4_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_5<=0,n_5<=PSI_TABLE_NMAX]
    phiNames = [None,None,ser_2]
    ser_3= phiIf(phiPreds, phiNames)
    phiPreds = [n_5<=0,n_5<=PSI_TABLE_NMAX]
    phiNames = [None,None,c5_0]
    c5_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_5<=0,n_5<=PSI_TABLE_NMAX]
    phiNames = [None,result_val_IV_24,result_val_IV_25]
    result_val_IV_26= phiIf(phiPreds, phiNames)
    phiPreds = [n_5<=0,n_5<=PSI_TABLE_NMAX]
    phiNames = [None,None,ni2_2]
    ni2_3= phiIf(phiPreds, phiNames)
    phiPreds = [n_5<=0,n_5<=PSI_TABLE_NMAX]
    phiNames = [None,psi_table_n_IV_0,None]
    psi_table_n_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [n_5<=0,n_5<=PSI_TABLE_NMAX]
    phiNames = [None,result_err_51,result_err_54]
    result_err_55= phiIf(phiPreds, phiNames)
    phiPreds = [n_5<=0,n_5<=PSI_TABLE_NMAX]
    phiNames = [None,None,c2_2]
    c2_3= phiIf(phiPreds, phiNames)

def gsl_sf_lnfact_e(n,result):
    n_6 = n;result_11 = result;
    result_val_37=None;result_val_38=None;result_val_IV_27=None;result_val_IV_28=None;result_err_56=None;result_err_57=None;fact_table_n_f_IV_0=None;fact_table_n_f_IV_1=None;

    if n_6<=GSL_SF_FACT_NMAX:
        fact_table_n_f_IV_0=fact_table[n_6].f 
        result_val_37=log(fact_table_n_f_IV_0) 
        result_11.val=result_val_37 
        result_val_IV_27=result_11.val 
        result_err_56=2.0*GSL_DBL_EPSILON*fabs(result_val_IV_27) 
        result_11.err=result_err_56 
        lo = locals()
        record_locals(lo, test_counter) 
        return GSL_SUCCESS
    else:
        gsl_sf_lngamma_e(n_6+1.0,result_11) 
        lo = locals()
        record_locals(lo, test_counter) 
        return GSL_SUCCESS
    phiPreds = [n_6<=GSL_SF_FACT_NMAX]
    phiNames = [result_val_37,None]
    result_val_38= phiIf(phiPreds, phiNames)
    phiPreds = [n_6<=GSL_SF_FACT_NMAX]
    phiNames = [result_val_IV_27,None]
    result_val_IV_28= phiIf(phiPreds, phiNames)
    phiPreds = [n_6<=GSL_SF_FACT_NMAX]
    phiNames = [result_err_56,None]
    result_err_57= phiIf(phiPreds, phiNames)
    phiPreds = [n_6<=GSL_SF_FACT_NMAX]
    phiNames = [fact_table_n_f_IV_0,None]
    fact_table_n_f_IV_1= phiIf(phiPreds, phiNames)

def lngamma_sgn_sing(N,eps,lng,sgn):
    N_3 = N;eps_0 = eps;lng_0 = lng;sgn_0 = sgn;
    cs1_0=None;cs1_1=None;psi_6_val_IV_0=None;psi_6_val_IV_1=None;cs3_0=None;cs3_1=None;psi_5_val_IV_0=None;psi_5_val_IV_1=None;cs2_0=None;cs2_1=None;cs5_0=None;cs5_1=None;psi_4_val_IV_0=None;psi_4_val_IV_1=None;cs4_0=None;cs4_1=None;aeps_0=None;aeps_1=None;c0_err_0=None;c0_err_1=None;psi_0_val_IV_0=None;psi_0_val_IV_1=None;psi_0_0=None;psi_0_1=None;psi_1_0=None;psi_1_1=None;psi_2_val_IV_0=None;psi_2_val_IV_1=None;psi_2_0=None;psi_2_1=None;c0_val_IV_0=None;c0_val_IV_1=None;psi_3_0=None;psi_3_1=None;psi_1_val_IV_0=None;psi_1_val_IV_1=None;psi_3_val_IV_0=None;psi_3_val_IV_1=None;sgn_1=None;sgn_2=None;sgn_3=None;sgn_4=None;psi_4_0=None;psi_4_1=None;psi_5_0=None;psi_5_1=None;psi_6_0=None;psi_6_1=None;gam_e_0=None;gam_e_1=None;lng_err_0=None;lng_err_1=None;lng_err_2=None;g_0=None;g_1=None;g_2=None;lng_ser_0=None;lng_ser_1=None;lng_val_0=None;lng_val_1=None;lng_val_2=None;c0_2=None;c0_3=None;c0_4=None;e2_0=None;e2_1=None;sin_ser_0=None;sin_ser_1=None;c1_2=None;c1_3=None;c1_4=None;g5_0=None;g5_1=None;c2_4=None;c2_5=None;c2_6=None;c3_2=None;c3_3=None;c3_4=None;c4_2=None;c4_3=None;c4_4=None;c5_2=None;c5_3=None;c5_4=None;c6_0=None;c6_1=None;c6_2=None;c7_0=None;c7_1=None;c7_2=None;lng_val_IV_0=None;lng_val_IV_1=None;lng_val_IV_2=None;c8_0=None;c8_1=None;c9_0=None;c9_1=None;

    if eps_0==0:
        lng_0.val=0.0 
        lng_0.val=0.0 
        sgn_1=0.0 
        print("edom error") 
    elif N_3==1:
        c0_2=0.07721566490153286061 
        c1_2=0.08815966957356030521 
        c2_4=-0.00436125434555340577 
        c3_2=0.01391065882004640689 
        c4_2=-0.00409427227680839100 
        c5_2=0.00275661310191541584 
        c6_0=-0.00124162645565305019 
        c7_0=0.00065267976121802783 
        c8_0=-0.00032205261682710437 
        c9_0=0.00016229131039545456 
        g5_0=c5_2+eps_0*(c6_0+eps_0*(c7_0+eps_0*(c8_0+eps_0*c9_0))) 
        g_0=eps_0*(c0_2+eps_0*(c1_2+eps_0*(c2_4+eps_0*(c3_2+eps_0*(c4_2+eps_0*g5_0))))) 
        gam_e_0=g_0-1.0-0.5*eps_0*(1.0+3.0*eps_0)/(1.0-eps_0*eps_0) 
        lng_val_0=log(fabs(gam_e_0)/fabs(eps_0)) 
        lng_0.val=lng_val_0 
        lng_val_IV_0=lng_0.val 
        lng_err_0=2.0*GSL_DBL_EPSILON*fabs(lng_val_IV_0) 
        lng_0.err=lng_err_0 
        sgn_2=-1.0 if eps_0>0.0 else 1.0
        lo = locals()
        record_locals(lo, test_counter)  
        return GSL_SUCCESS
    else:
        cs1_0=-1.6449340668482264365 
        cs2_0=0.8117424252833536436 
        cs3_0=-0.1907518241220842137 
        cs4_0=0.0261478478176548005 
        cs5_0=-0.0023460810354558236 
        e2_0=eps_0*eps_0 
        sin_ser_0=1.0+e2_0*(cs1_0+e2_0*(cs2_0+e2_0*(cs3_0+e2_0*(cs4_0+e2_0*cs5_0)))) 
        aeps_0=fabs(eps_0) 
        c0_3=gsl_sf_result(0,0) 
        psi_0_0=gsl_sf_result(0,0) 
        psi_1_0=gsl_sf_result(0,0) 
        psi_2_0=gsl_sf_result(0,0) 
        psi_3_0=gsl_sf_result(0,0) 
        psi_4_0=gsl_sf_result(0,0) 
        psi_5_0=gsl_sf_result(0,0) 
        psi_6_0=gsl_sf_result(0,0) 
        psi_2_0.val=0.0 
        psi_3_0.val=0.0 
        psi_4_0.val=0.0 
        psi_5_0.val=0.0 
        psi_6_0.val=0.0 
        gsl_sf_lnfact_e(N_3,c0_3) 
        gsl_sf_psi_int_e(N_3+1,psi_0_0) 
        gsl_sf_psi_1_int_e(N_3+1,psi_1_0) 
        if aeps_0>0.00001:
            gsl_sf_psi_n_e(2,N_3+1.0,psi_2_0) 
        if aeps_0>0.0002:
            gsl_sf_psi_n_e(3,N_3+1.0,psi_3_0) 
        if aeps_0>0.001:
            gsl_sf_psi_n_e(4,N_3+1.0,psi_4_0) 
        if aeps_0>0.005:
            gsl_sf_psi_n_e(5,N_3+1.0,psi_5_0) 
        if aeps_0>0.01:
            gsl_sf_psi_n_e(6,N_3+1.0,psi_6_0) 
        psi_0_val_IV_0=psi_0_0.val 
        c1_3=psi_0_val_IV_0 
        psi_1_val_IV_0=psi_1_0.val 
        c2_5=psi_1_val_IV_0/2.0 
        psi_2_val_IV_0=psi_2_0.val 
        c3_3=psi_2_val_IV_0/6.0 
        psi_3_val_IV_0=psi_3_0.val 
        c4_3=psi_3_val_IV_0/24.0 
        psi_4_val_IV_0=psi_4_0.val 
        c5_3=psi_4_val_IV_0/120.0 
        psi_5_val_IV_0=psi_5_0.val 
        c6_1=psi_5_val_IV_0/720.0 
        psi_6_val_IV_0=psi_6_0.val 
        c7_1=psi_6_val_IV_0/5040.0 
        c0_val_IV_0=c0_3.val 
        lng_ser_0=c0_val_IV_0-eps_0*(c1_3-eps_0*(c2_5-eps_0*(c3_3-eps_0*(c4_3-eps_0*(c5_3-eps_0*(c6_1-eps_0*c7_1)))))) 
        g_1=-lng_ser_0-log(sin_ser_0) 
        lng_val_1=g_1-log(fabs(eps_0)) 
        lng_0.val=lng_val_1 
        lng_val_IV_1=lng_0.val 
        c0_err_0=c0_3.err 
        lng_err_1=c0_err_0+2.0*GSL_DBL_EPSILON*(fabs(g_1)+fabs(lng_val_IV_1)) 
        lng_0.err=lng_err_1 
        sgn_3=(-1.0 if GSL_IS_ODD(N_3) else 1.0)*(1.0 if eps_0>0.0 else -1.0) 
        lo = locals()
        record_locals(lo, test_counter)  
        return GSL_SUCCESS
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,cs1_0]
    cs1_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,psi_6_val_IV_0]
    psi_6_val_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,cs3_0]
    cs3_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,psi_5_val_IV_0]
    psi_5_val_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,cs2_0]
    cs2_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,cs5_0]
    cs5_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,psi_4_val_IV_0]
    psi_4_val_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,cs4_0]
    cs4_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,aeps_0]
    aeps_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,c0_err_0]
    c0_err_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,psi_0_val_IV_0]
    psi_0_val_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,psi_0_0]
    psi_0_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,psi_1_0]
    psi_1_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,psi_2_val_IV_0]
    psi_2_val_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,psi_2_0]
    psi_2_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,c0_val_IV_0]
    c0_val_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,psi_3_0]
    psi_3_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,psi_1_val_IV_0]
    psi_1_val_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,psi_3_val_IV_0]
    psi_3_val_IV_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [sgn_1,sgn_2,sgn_3]
    sgn_4= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,psi_4_0]
    psi_4_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,psi_5_0]
    psi_5_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,psi_6_0]
    psi_6_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,gam_e_0,None]
    gam_e_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,lng_err_0,lng_err_1]
    lng_err_2= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,g_0,g_1]
    g_2= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,lng_ser_0]
    lng_ser_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,lng_val_0,lng_val_1]
    lng_val_2= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,c0_2,c0_3]
    c0_4= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,e2_0]
    e2_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,None,sin_ser_0]
    sin_ser_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,c1_2,c1_3]
    c1_4= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,g5_0,None]
    g5_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,c2_4,c2_5]
    c2_6= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,c3_2,c3_3]
    c3_4= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,c4_2,c4_3]
    c4_4= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,c5_2,c5_3]
    c5_4= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,c6_0,c6_1]
    c6_2= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,c7_0,c7_1]
    c7_2= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,lng_val_IV_0,lng_val_IV_1]
    lng_val_IV_2= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,c8_0,None]
    c8_1= phiIf(phiPreds, phiNames)
    phiPreds = [eps_0==0,N_3==1]
    phiNames = [None,c9_0,None]
    c9_1= phiIf(phiPreds, phiNames)

def lngamma_1_pade(eps,result):
    eps_1 = eps;result_12 = result;
    eps5_0=None;corr_0=None;n1_0=None;n2_0=None;pade_0=None;num_0=None;den_0=None;result_err_58=None;d1_0=None;c0_5=None;d2_0=None;c1_5=None;c2_7=None;c3_5=None;result_val_39=None;c4_5=None;result_val_IV_29=None;

    n1_0=-1.0017419282349508699871138440 
    n2_0=1.7364839209922879823280541733 
    d1_0=1.2433006018858751556055436011 
    d2_0=5.0456274100274010152489597514 
    num_0=(eps_1+n1_0)*(eps_1+n2_0) 
    den_0=(eps_1+d1_0)*(eps_1+d2_0) 
    pade_0=2.0816265188662692474880210318*num_0/den_0 
    c0_5=0.004785324257581753 
    c1_5=-0.01192457083645441 
    c2_7=0.01931961413960498 
    c3_5=-0.02594027398725020 
    c4_5=0.03141928755021455 
    eps5_0=eps_1*eps_1*eps_1*eps_1*eps_1 
    corr_0=eps5_0*(c0_5+eps_1*(c1_5+eps_1*(c2_7+eps_1*(c3_5+c4_5*eps_1)))) 
    result_val_39=eps_1*(pade_0+corr_0) 
    result_12.val=result_val_39 
    result_val_IV_29=result_12.val 
    result_err_58=2.0*GSL_DBL_EPSILON*fabs(result_val_IV_29) 
    result_12.err=result_err_58 
    lo = locals()
    record_locals(lo, test_counter)  
    return GSL_SUCCESS

def lngamma_2_pade(eps,result):
    eps_2 = eps;result_13 = result;
    eps5_1=None;corr_1=None;n1_1=None;n2_1=None;pade_1=None;num_1=None;den_1=None;result_err_59=None;d1_1=None;c0_6=None;d2_1=None;c1_6=None;c2_8=None;c3_6=None;result_val_40=None;c4_6=None;result_val_IV_30=None;

    n1_1=1.000895834786669227164446568 
    n2_1=4.209376735287755081642901277 
    d1_1=2.618851904903217274682578255 
    d2_1=10.85766559900983515322922936 
    num_1=(eps_2+n1_1)*(eps_2+n2_1) 
    den_1=(eps_2+d1_1)*(eps_2+d2_1) 
    pade_1=2.85337998765781918463568869*num_1/den_1 
    c0_6=0.0001139406357036744 
    c1_6=-0.0001365435269792533 
    c2_8=0.0001067287169183665 
    c3_6=-0.0000693271800931282 
    c4_6=0.0000407220927867950 
    eps5_1=eps_2*eps_2*eps_2*eps_2*eps_2 
    corr_1=eps5_1*(c0_6+eps_2*(c1_6+eps_2*(c2_8+eps_2*(c3_6+c4_6*eps_2)))) 
    result_val_40=eps_2*(pade_1+corr_1) 
    result_13.val=result_val_40 
    result_val_IV_30=result_13.val 
    result_err_59=2.0*GSL_DBL_EPSILON*fabs(result_val_IV_30) 
    result_13.err=result_err_59 
    lo = locals()
    record_locals(lo, test_counter)  
    return GSL_SUCCESS

lanczos_7_c=[0.99999999999980993227684700473478,676.520368121885098567009190444019,-1259.13921672240287047156078755283,771.3234287776530788486528258894,-176.61502916214059906584551354,12.507343278686904814458936853,-0.13857109526572011689554707,9.984369578019570859563e-6,1.50563273514931155834e-7] 
def lngamma_lanczos(x,result):
    x_8 = x;result_14 = result;
    result_val_41=None;term2_0=None;result_val_IV_31=None;Ag_0=None;Ag_2=None;Ag_1=None;Ag_3=None;lanczos_7_c_k_IV_1=None;lanczos_7_c_k_IV_0=None;lanczos_7_c_k_IV_2=None;term1_0=None;x_9=None;result_err_60=None;result_err_61=None;result_err_62=None;

    x_9 = x_8-1.0
    Ag_0=lanczos_7_c[0] 
    phi0 = Phi()
    for k_1 in range(1,9):
        phi0.set()
        Ag_2 = phi0.phiEntry(Ag_0,Ag_1)
        lanczos_7_c_k_IV_1 = phi0.phiEntry(None,lanczos_7_c_k_IV_0)

        lanczos_7_c_k_IV_0=lanczos_7_c[k_1] 
        Ag_1 = Ag_2+lanczos_7_c_k_IV_0/(x_9+k_1) + bug
    Ag_3 = phi0.phiExit(Ag_0,Ag_1)
    lanczos_7_c_k_IV_2 = phi0.phiExit(None,lanczos_7_c_k_IV_0)
    term1_0=(x_9+0.5)*log((x_9+7.5)/M_E)
    term2_0=LogRootTwoPi_+log(Ag_3) 
    result_val_41=term1_0+(term2_0-7.0) 
    result_14.val=result_val_41 
    result_err_60=2.0*GSL_DBL_EPSILON*(fabs(term1_0)+fabs(term2_0)+7.0) 
    result_14.err=result_err_60 
    result_val_IV_31=result_14.val 
    result_err_61=result_14.err 
    result_err_62 = result_err_61+GSL_DBL_EPSILON*fabs(result_val_IV_31)
    result_14.err=result_err_62
    lo = locals()
    record_locals(lo, test_counter)   
    return GSL_SUCCESS

def lngamma_sgn_0(eps,lgn,sgn):
    eps_3 = eps;lgn_0 = lgn;sgn_5 = sgn;
    c10_0=None;gee_0=None;g_3=None;lgn_err_0=None;lgn_val_0=None;lgn_val_IV_0=None;c1_7=None;c2_9=None;g6_0=None;c3_7=None;c4_7=None;c5_5=None;c6_3=None;c7_3=None;c8_2=None;c9_2=None;sgn_6=None;

    c1_7=-0.07721566490153286061 
    c2_9=-0.01094400467202744461 
    c3_7=0.09252092391911371098 
    c4_7=-0.01827191316559981266 
    c5_5=0.01800493109685479790 
    c6_3=-0.00685088537872380685 
    c7_3=0.00399823955756846603 
    c8_2=-0.00189430621687107802 
    c9_2=0.00097473237804513221 
    c10_0=-0.00048434392722255893 
    g6_0=c6_3+eps_3*(c7_3+eps_3*(c8_2+eps_3*(c9_2+eps_3*c10_0))) 
    g_3=eps_3*(c1_7+eps_3*(c2_9+eps_3*(c3_7+eps_3*(c4_7+eps_3*(c5_5+eps_3*g6_0))))) 
    gee_0=g_3+1.0/(1.0+eps_3)+0.5*eps_3 
    lgn_val_0=log(gee_0/fabs(eps_3)) 
    lgn_0.val=lgn_val_0 
    lgn_val_IV_0=lgn_0.val 
    lgn_err_0=4.0*GSL_DBL_EPSILON*fabs(lgn_val_IV_0) 
    lgn_0.err=lgn_err_0 
    sgn_6=GSL_SIGN(eps_3) 
    lo = locals()
    record_locals(lo, test_counter) 
    return GSL_SUCCESS

def gsl_sf_lngamma_e(x,result):
    x_10 = x;result_15 = result;
    stat_0=None;stat_1=None;stat_2=None;lg_z_0=None;lg_z_1=None;lg_z_2=None;lg_z_val_IV_0=None;lg_z_val_IV_1=None;lg_z_val_IV_2=None;eps_4=None;eps_5=None;eps_6=None;eps_7=None;return_val_0=None;return_val_1=None;return_val_2=None;return_val_3=None;return_val_4=None;return_val_5=None;result_err_63=None;result_err_64=None;result_err_65=None;result_err_66=None;result_err_67=None;result_err_68=None;result_err_69=None;result_err_70=None;result_err_71=None;result_err_72=None;N_4=None;N_5=None;N_6=None;N_7=None;_as_0=None;_as_1=None;result_val_42=None;result_val_43=None;result_val_44=None;result_val_45=None;result_val_46=None;result_val_47=None;s_4=None;s_5=None;result_val_IV_32=None;result_val_IV_33=None;result_val_IV_34=None;z_0=None;z_1=None;sgn_7=None;sgn_8=None;sgn_9=None;sgn_10=None;sgn_11=None;lg_z_err_0=None;lg_z_err_1=None;lg_z_err_2=None;

    if fabs(x_10-1.0)<0.01:
        stat_0=lngamma_1_pade(x_10-1.0,result_15) 
        result_err_63=result_15.err 
        result_err_64 = result_err_63*1.0/(GSL_DBL_EPSILON+fabs(x_10-1.0))
        result_15.err=result_err_64 
        lo = locals()
        record_locals(lo, test_counter) 
        return stat_0
    elif fabs(x_10-2.0)<0.01:
        stat_1=lngamma_2_pade(x_10-2.0,result_15) 
        result_err_65=result_15.err 
        result_err_66 = result_err_65*1.0/(GSL_DBL_EPSILON+fabs(x_10-2.0))
        result_15.err=result_err_66 
        lo = locals()
        record_locals(lo, test_counter) 
        return stat_1
    elif x_10>=0.5:
        return_val_0=lngamma_lanczos(x_10,result_15) 
        lo = locals()
        record_locals(lo, test_counter) 
        return return_val_0
    elif x_10==0.0:
        print("domain error") 
    elif fabs(x_10)<0.02:
        sgn_7=0.0 
        return_val_1=lngamma_sgn_0(x_10,result_15,sgn_7) 
        lo = locals()
        record_locals(lo, test_counter) 
        return return_val_1
    elif x_10>-0.5/(GSL_DBL_EPSILON*M_PI):
        z_0=1.0-x_10 
        s_4=sin(M_PI*z_0) 
        _as_0=fabs(s_4) 
        if s_4==0.0:
            print("domain error") 
        elif _as_0<M_PI*0.015:
            if x_10<float('-inf')+2.0:
                result_val_42=0.0 
                result_15.val=result_val_42 
                result_err_67=0.0 
                result_15.err=result_err_67 
                print("eround error") 
            else:
                N_4=-(x_10-0.5) 
                eps_4=x_10+N_4 
                sgn_8=0 
                return_val_2=lngamma_sgn_sing(N_4,eps_4,result_15,sgn_8) 
                lo = locals()
                record_locals(lo, test_counter) 
                return return_val_2
            phiPreds = [x_10<float('-inf')+2.0]
            phiNames = [result_val_42,None]
            result_val_43= phiIf(phiPreds, phiNames)
            phiPreds = [x_10<float('-inf')+2.0]
            phiNames = [None,eps_4]
            eps_5= phiIf(phiPreds, phiNames)
            phiPreds = [x_10<float('-inf')+2.0]
            phiNames = [None,sgn_8]
            sgn_9= phiIf(phiPreds, phiNames)
            phiPreds = [x_10<float('-inf')+2.0]
            phiNames = [None,return_val_2]
            return_val_3= phiIf(phiPreds, phiNames)
            phiPreds = [x_10<float('-inf')+2.0]
            phiNames = [result_err_67,None]
            result_err_68= phiIf(phiPreds, phiNames)
            phiPreds = [x_10<float('-inf')+2.0]
            phiNames = [None,N_4]
            N_5= phiIf(phiPreds, phiNames)
        else:
            lg_z_0=gsl_sf_result(0,0) 
            lngamma_lanczos(z_0,lg_z_0) 
            lg_z_val_IV_0=lg_z_0.val 
            result_val_44=M_LNPI-(log(_as_0)+lg_z_val_IV_0) 
            result_15.val=result_val_44 
            result_val_IV_32=result_15.val 
            lg_z_err_0=lg_z_0.err 
            result_err_69=2.0*GSL_DBL_EPSILON*fabs(result_val_IV_32)+lg_z_err_0 
            result_15.err=result_err_69 
            lo = locals()
            record_locals(lo, test_counter) 
            return GSL_SUCCESS
        phiPreds = [s_4==0.0,_as_0<M_PI*0.015]
        phiNames = [None,result_val_43,result_val_44]
        result_val_45= phiIf(phiPreds, phiNames)
        phiPreds = [s_4==0.0,_as_0<M_PI*0.015]
        phiNames = [None,None,lg_z_0]
        lg_z_1= phiIf(phiPreds, phiNames)
        phiPreds = [s_4==0.0,_as_0<M_PI*0.015]
        phiNames = [None,None,lg_z_val_IV_0]
        lg_z_val_IV_1= phiIf(phiPreds, phiNames)
        phiPreds = [s_4==0.0,_as_0<M_PI*0.015]
        phiNames = [None,None,result_val_IV_32]
        result_val_IV_33= phiIf(phiPreds, phiNames)
        phiPreds = [s_4==0.0,_as_0<M_PI*0.015]
        phiNames = [None,eps_5,None]
        eps_6= phiIf(phiPreds, phiNames)
        phiPreds = [s_4==0.0,_as_0<M_PI*0.015]
        phiNames = [None,sgn_9,None]
        sgn_10= phiIf(phiPreds, phiNames)
        phiPreds = [s_4==0.0,_as_0<M_PI*0.015]
        phiNames = [None,return_val_3,None]
        return_val_4= phiIf(phiPreds, phiNames)
        phiPreds = [s_4==0.0,_as_0<M_PI*0.015]
        phiNames = [None,result_err_68,result_err_69]
        result_err_70= phiIf(phiPreds, phiNames)
        phiPreds = [s_4==0.0,_as_0<M_PI*0.015]
        phiNames = [None,None,lg_z_err_0]
        lg_z_err_1= phiIf(phiPreds, phiNames)
        phiPreds = [s_4==0.0,_as_0<M_PI*0.015]
        phiNames = [None,N_5,None]
        N_6= phiIf(phiPreds, phiNames)
    else:
        result_val_46=0.0 
        result_15.val=result_val_46 
        result_err_71=0.0 
        result_15.err=result_err_71 
        print("eround error") 
    phiPreds = [fabs(x_10-1.0)<0.01,fabs(x_10-2.0)<0.01,x_10>=0.5,x_10==0.0,fabs(x_10)<0.02,x_10>-0.5/(GSL_DBL_EPSILON*M_PI)]
    phiNames = [stat_0,stat_1,None,None,None,None,None]
    stat_2= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(x_10-1.0)<0.01,fabs(x_10-2.0)<0.01,x_10>=0.5,x_10==0.0,fabs(x_10)<0.02,x_10>-0.5/(GSL_DBL_EPSILON*M_PI)]
    phiNames = [None,None,None,None,None,lg_z_1,None]
    lg_z_2= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(x_10-1.0)<0.01,fabs(x_10-2.0)<0.01,x_10>=0.5,x_10==0.0,fabs(x_10)<0.02,x_10>-0.5/(GSL_DBL_EPSILON*M_PI)]
    phiNames = [None,None,None,None,None,lg_z_val_IV_1,None]
    lg_z_val_IV_2= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(x_10-1.0)<0.01,fabs(x_10-2.0)<0.01,x_10>=0.5,x_10==0.0,fabs(x_10)<0.02,x_10>-0.5/(GSL_DBL_EPSILON*M_PI)]
    phiNames = [None,None,None,None,None,eps_6,None]
    eps_7= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(x_10-1.0)<0.01,fabs(x_10-2.0)<0.01,x_10>=0.5,x_10==0.0,fabs(x_10)<0.02,x_10>-0.5/(GSL_DBL_EPSILON*M_PI)]
    phiNames = [None,None,return_val_0,None,return_val_1,return_val_4,None]
    return_val_5= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(x_10-1.0)<0.01,fabs(x_10-2.0)<0.01,x_10>=0.5,x_10==0.0,fabs(x_10)<0.02,x_10>-0.5/(GSL_DBL_EPSILON*M_PI)]
    phiNames = [result_err_64,result_err_66,None,None,None,result_err_70,result_err_71]
    result_err_72= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(x_10-1.0)<0.01,fabs(x_10-2.0)<0.01,x_10>=0.5,x_10==0.0,fabs(x_10)<0.02,x_10>-0.5/(GSL_DBL_EPSILON*M_PI)]
    phiNames = [None,None,None,None,None,N_6,None]
    N_7= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(x_10-1.0)<0.01,fabs(x_10-2.0)<0.01,x_10>=0.5,x_10==0.0,fabs(x_10)<0.02,x_10>-0.5/(GSL_DBL_EPSILON*M_PI)]
    phiNames = [None,None,None,None,None,_as_0,None]
    _as_1= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(x_10-1.0)<0.01,fabs(x_10-2.0)<0.01,x_10>=0.5,x_10==0.0,fabs(x_10)<0.02,x_10>-0.5/(GSL_DBL_EPSILON*M_PI)]
    phiNames = [None,None,None,None,None,result_val_45,result_val_46]
    result_val_47= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(x_10-1.0)<0.01,fabs(x_10-2.0)<0.01,x_10>=0.5,x_10==0.0,fabs(x_10)<0.02,x_10>-0.5/(GSL_DBL_EPSILON*M_PI)]
    phiNames = [None,None,None,None,None,s_4,None]
    s_5= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(x_10-1.0)<0.01,fabs(x_10-2.0)<0.01,x_10>=0.5,x_10==0.0,fabs(x_10)<0.02,x_10>-0.5/(GSL_DBL_EPSILON*M_PI)]
    phiNames = [None,None,None,None,None,result_val_IV_33,None]
    result_val_IV_34= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(x_10-1.0)<0.01,fabs(x_10-2.0)<0.01,x_10>=0.5,x_10==0.0,fabs(x_10)<0.02,x_10>-0.5/(GSL_DBL_EPSILON*M_PI)]
    phiNames = [None,None,None,None,None,z_0,None]
    z_1= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(x_10-1.0)<0.01,fabs(x_10-2.0)<0.01,x_10>=0.5,x_10==0.0,fabs(x_10)<0.02,x_10>-0.5/(GSL_DBL_EPSILON*M_PI)]
    phiNames = [None,None,None,None,sgn_7,sgn_10,None]
    sgn_11= phiIf(phiPreds, phiNames)
    phiPreds = [fabs(x_10-1.0)<0.01,fabs(x_10-2.0)<0.01,x_10>=0.5,x_10==0.0,fabs(x_10)<0.02,x_10>-0.5/(GSL_DBL_EPSILON*M_PI)]
    phiNames = [None,None,None,None,None,lg_z_err_1,None]
    lg_z_err_2= phiIf(phiPreds, phiNames)

def gsl_sf_lngamma(x):
    x_11 = x;
    result_16=None;

    result_16=gsl_sf_result(0.0,0.0) 
    return EVAL_RESULT(gsl_sf_lngamma_e(x_11,result_16),result_16)



#generate python causal map
causal_map = {'psi_0_0':[],'psi_0_1':['psi_0_0'],'n2_1':[],'c2_9':[],'c2_8':[],'n2_0':[],'cs_c_j_IV_1':['cs_c_j_IV_0'],'c2_5':['psi_1_val_IV_0'],'cs_c_j_IV_0':['cs_0','j_0'],'c2_4':[],'c2_7':[],'c2_6':['c2_4','c2_5'],'c2_1':['c2_0'],'c2_0':[],'c2_3':['c2_2'],'c2_2':[],'dd_3':['dd_0','dd_1'],'dd_1':['temp_0'],'dd_2':['dd_0','dd_1'],'dd_0':[],'cs_c_cs_order_IV_0':['cs_0','cs_0'],'stat_hz_0':['n_2','x_5','hzeta_0'],'stat_hz_1':['stat_hz_0'],'stat_hz_2':['n_3','x_7','hzeta_2'],'stat_hz_3':['stat_hz_2'],'fx_0':['x_6','M_3'],'fx_1':['fx_0'],'cs_c_j_IV_2':['cs_c_j_IV_0'],'result_val_44':['_as_0','lg_z_val_IV_0'],'result_val_45':['result_val_43','result_val_44'],'result_val_46':[],'result_val_47':['result_val_45','result_val_46'],'result_val_5':['result_val_3','result_val_4'],'psi_1_1':['psi_1_0'],'result_val_4':['y_1','x_2','result_c_val_IV_2'],'result_val_7':['t1_1','t2_1','result_c_val_IV_5'],'result_val_6':['t1_0','t2_0','t3_0','result_c_val_IV_4'],'psi_1_0':[],'result_val_1':['d_4'],'result_val_40':['eps_2','pade_1','corr_1'],'result_val_0':['result_0'],'result_val_41':['term1_0','term2_0'],'result_val_3':['result_val_2'],'n1_0':[],'result_val_42':[],'result_val_2':['y_1','x_2','result_c_val_IV_0','c_2','s_0'],'n1_1':[],'result_val_43':['result_val_42'],'c1_6':[],'c1_5':[],'c1_7':[],'result_val_9':['result_val_6','result_val_7','result_val_8'],'c1_2':[],'sin_ser_1':['sin_ser_0'],'result_val_8':['t1_2','result_c_val_IV_6'],'c1_1':['c1_0'],'sin_ser_0':['e2_0','cs1_0','e2_0','cs2_0','e2_0','cs3_0','e2_0','cs4_0','e2_0','cs5_0'],'c1_4':['c1_2','c1_3'],'c1_3':['psi_0_val_IV_0'],'aeps_1':['aeps_0'],'aeps_0':['eps_0'],'d2_0':[],'term2_0':['Ag_3'],'ln_nf_err_IV_0':['ln_nf_0'],'result_val_33':['result_val_31','result_val_32'],'ln_nf_err_IV_1':['ln_nf_err_IV_0'],'result_val_34':['psi_table_n_IV_0'],'result_val_35':['n_5','n_5','ser_2'],'result_val_36':['result_val_34','result_val_35'],'result_val_37':['fact_table_n_f_IV_0'],'result_val_38':['result_val_37'],'ln_nf_err_IV_2':['ln_nf_2'],'result_val_39':['eps_1','pade_0','corr_0'],'ln_nf_err_IV_3':['ln_nf_err_IV_2'],'result_val_30':['result_val_29'],'result_val_31':['psi_1_table_n_IV_0'],'result_val_32':['n_4','n_4','n_4','ser_0','n_4'],'c1_0':[],'result_val_22':['result_val_21'],'result_val_23':['result_val_22'],'result_val_24':['result_7'],'result_val_25':['result_val_24','sum_3'],'result_val_26':['d_5','r_val_IV_0'],'y2_0':['y_0'],'result_val_27':['result_val_25','result_val_26'],'N_0':['ly_0'],'result_val_28':['result_val_IV_18'],'result_val_29':['result_val_28'],'N_2':['N_1'],'psi_2_0':[],'N_1':['N_0'],'psi_2_1':['psi_2_0'],'N_4':['x_10'],'N_6':['N_5'],'p3_2':['p3_1'],'N_5':['N_4'],'p3_1':['p3_0'],'result_val_20':['result_val_16','result_val_17','result_val_19'],'p3_0':['q_0','q_0','s_3'],'result_val_21':['result_val_IV_15'],'N_7':['N_6'],'c0_6':[],'c0_3':[],'c0_2':[],'c0_5':[],'c0_4':['c0_2','c0_3'],'g6_0':['c6_3','eps_3','c7_3','eps_3','c8_2','eps_3','c9_2','eps_3','c10_0'],'result_val_19':['result_val_18'],'d1_0':[],'d1_1':[],'term1_0':['x_9','x_9'],'result_val_11':['q_0','s_3'],'result_val_12':['p1_0','p2_0','p3_0'],'result_val_13':['ans_6'],'cs1_0':[],'result_val_14':['result_val_11','result_val_12','result_val_13'],'cs1_1':['cs1_0'],'result_val_15':['result_val_14'],'result_val_16':[],'result_val_17':['y_2','ex_0'],'result_val_18':['sy_0','eMN_0','eab_0'],'sy_0':['y_2'],'result_val_10':['result_val_5','result_val_9'],'cs_b_IV_0':['cs_0'],'sy_1':['sy_0'],'d2_1':[],'sy_2':['sy_1'],'c0_1':['c0_0'],'c0_0':[],'result_err_70':['result_err_68','result_err_69'],'M_1':['M_0'],'result_err_71':[],'M_0':['x_3'],'result_err_72':['result_err_64','result_err_66','result_err_70','result_err_71'],'p2_0':['q_0','q_0','s_3'],'M_3':['x_6'],'M_2':['M_1'],'psi_3_0':[],'M_4':['M_3'],'lnr_0':['x_3','ly_0'],'eab_1':['eab_0'],'eab_2':['eab_1'],'p2_2':['p2_1'],'psi_3_1':['psi_3_0'],'p2_1':['p2_0'],'lnr_1':['lnr_0'],'result_err_59':['result_val_IV_30'],'eab_0':['a_3','b_3'],'g5_1':['g5_0'],'result_err_62':['result_err_61','result_val_IV_31'],'result_err_63':['result_15'],'result_err_64':['result_err_63','x_10'],'result_err_65':['result_15'],'result_err_66':['result_err_65','x_10'],'lg_z_val_IV_2':['lg_z_val_IV_1'],'result_err_67':[],'lg_z_val_IV_1':['lg_z_val_IV_0'],'result_err_68':['result_err_67'],'lg_z_val_IV_0':['lg_z_0'],'result_err_69':['result_val_IV_32','lg_z_err_0'],'result_err_60':['term1_0','term2_0'],'result_err_61':['result_14'],'stat_e_3':['stat_e_2'],'stat_e_2':['ln_nf_val_IV_2','ln_nf_err_IV_2','hzeta_val_IV_2','hzeta_err_IV_2','result_8'],'gee_0':['g_3','eps_3','eps_3'],'stat_e_1':['stat_e_0'],'stat_e_0':['ln_nf_val_IV_0','ln_nf_err_IV_0','hzeta_val_IV_0','hzeta_err_IV_0','result_6'],'result_err_48':['result_val_IV_21'],'pmax_0':['kmax_0','q_0','s_3'],'result_err_49':['result_val_IV_22'],'pmax_1':['pmax_0'],'pmax_2':['pmax_1'],'result_err_51':['result_val_IV_24'],'g5_0':['c5_2','eps_0','c6_0','eps_0','c7_0','eps_0','c8_0','eps_0','c9_0'],'e2_0':['eps_0','eps_0'],'result_err_52':['n_5','n_5','ser_2'],'e2_1':['e2_0'],'result_err_53':['result_10'],'result_err_54':['result_err_53','result_val_IV_25'],'result_err_55':['result_err_51','result_err_54'],'psi_3_val_IV_0':['psi_3_0'],'result_err_56':['result_val_IV_27'],'result_err_57':['result_err_56'],'psi_3_val_IV_1':['psi_3_val_IV_0'],'result_err_58':['result_val_IV_29'],'result_err_50':['result_err_48','result_err_49'],'p1_1':['p1_0'],'lng_val_IV_2':['lng_val_IV_0','lng_val_IV_1'],'p1_0':['q_0','s_3'],'cs_c_0_IV_0':['cs_0'],'p1_2':['p1_1'],'lng_val_IV_0':['lng_0'],'lng_val_IV_1':['lng_0'],'stat_nf_3':['stat_nf_2'],'stat_nf_2':['n_3','ln_nf_2'],'stat_nf_1':['stat_nf_0'],'stat_nf_0':['n_2','ln_nf_0'],'delta_0':['hzeta_c_j_1_IV_0','scp_2','pcp_2'],'delta_1':['delta_0'],'delta_2':['delta_0'],'delta_3':['delta_2'],'cs3_0':[],'delta_4':['delta_3'],'cs3_1':['cs3_0'],'_as_0':['s_4'],'_as_1':['_as_0'],'temp_0':['d_2'],'temp_1':['temp_0'],'temp_2':['temp_0'],'r_err_IV_0':['r_0'],'r_err_IV_1':['r_err_IV_0'],'temp_3':['d_3'],'hzeta_c_j_1_IV_3':['hzeta_c_j_1_IV_2'],'hzeta_c_j_1_IV_2':['hzeta_c_j_1_IV_0'],'hzeta_c_j_1_IV_1':['hzeta_c_j_1_IV_0'],'hzeta_c_j_1_IV_0':['j_1'],'scp_2':['scp_0','scp_1'],'scp_1':['scp_2','s_3','j_1','s_3','j_1'],'scp_0':['s_3'],'scp_5':['scp_4'],'scp_4':['scp_3'],'scp_3':['scp_0','scp_1'],'cs2_0':[],'cs2_1':['cs2_0'],'hzeta_c_j_1_IV_4':['hzeta_c_j_1_IV_3'],'c10_0':[],'c0_val_IV_1':['c0_val_IV_0'],'c0_val_IV_0':['c0_3'],'hzeta_3':['hzeta_2'],'lgn_val_0':['gee_0','eps_3'],'hzeta_0':[],'hzeta_1':['hzeta_0'],'psi_2_val_IV_0':['psi_2_0'],'hzeta_2':[],'psi_2_val_IV_1':['psi_2_val_IV_0'],'ans_3':['ans_0','ans_1'],'ans_4':['ans_5','delta_0'],'ans_1':['ans_2','k_0','q_0','s_3'],'ans_2':['ans_0','ans_1'],'ans_7':['ans_6'],'ans_8':['ans_7'],'ans_5':['ans_3','ans_4'],'ans_6':['ans_4'],'cs_a_IV_0':['cs_0'],'ans_0':['pmax_0','kmax_0','q_0','s_3'],'cs5_0':[],'cs5_1':['cs5_0'],'lg_z_err_0':['lg_z_0'],'lg_z_err_2':['lg_z_err_1'],'lg_z_err_1':['lg_z_err_0'],'z_0':['x_10'],'lgn_err_0':['lgn_val_IV_0'],'z_1':['z_0'],'sum_3':['sum_0','sum_1'],'sum_2':['sum_0','sum_1'],'sum_1':['sum_2','x_6','m_0','x_6','m_0'],'sum_0':[],'psi_1_table_n_IV_0':['n_4'],'psi_1_table_n_IV_1':['psi_1_table_n_IV_0'],'status_0':['fn_0'],'pade_1':['num_1','den_1'],'ay_0':['y_2'],'pade_0':['num_0','den_0'],'sum_4':['sum_3'],'ly_0':['ay_0'],'ly_1':['ly_0'],'den_0':['eps_1','d1_0','eps_1','d2_0'],'den_1':['eps_2','d1_1','eps_2','d2_1'],'result_c_2':['result_c_0','result_c_1'],'result_c_1':[],'result_c_0':[],'cs4_0':[],'cs4_1':['cs4_0'],'r_val_IV_0':['r_0'],'r_val_IV_1':['r_val_IV_0'],'y_1':['x_2'],'y_0':['x_1','cs_a_IV_0','cs_b_IV_0','cs_b_IV_0','cs_a_IV_0'],'lng_val_2':['lng_val_0','lng_val_1'],'lng_val_0':['gam_e_0','eps_0'],'lng_val_1':['g_1','eps_0'],'ni2_2':['n_5','n_5'],'ni2_3':['ni2_2'],'ni2_0':['n_4','n_4'],'ni2_1':['ni2_0'],'sgn_10':['sgn_9'],'sgn_11':['sgn_7','sgn_10'],'t3_1':['t3_0'],'t3_0':['v_0'],'t3_2':['t3_1'],'psi_5_val_IV_1':['psi_5_val_IV_0'],'result_16':[],'result_val_IV_13':['result_4'],'result_val_IV_14':['result_val_IV_13'],'result_val_IV_15':['result_6'],'result_val_IV_16':['result_val_IV_15'],'result_val_IV_10':['result_3'],'result_val_IV_11':['result_val_IV_9','result_val_IV_10'],'result_val_IV_12':['result_val_IV_11'],'result_val_IV_17':['result_val_IV_16'],'result_val_IV_18':['result_8'],'lng_err_2':['lng_err_0','lng_err_1'],'result_val_IV_19':['result_val_IV_18'],'lng_err_1':['c0_err_0','g_1','lng_val_IV_1'],'lng_err_0':['lng_val_IV_0'],'ln_term0_0':['s_3','q_0'],'ln_term0_1':['ln_term0_0'],'x_9':['x_8'],'psi_5_val_IV_0':['psi_5_0'],'psi_0_val_IV_1':['psi_0_val_IV_0'],'psi_0_val_IV_0':['psi_0_0'],'ln_nf_0':[],'ln_nf_1':['ln_nf_0'],'t2_2':['t2_0','t2_1'],'c9_2':[],'t2_1':['v_1'],'c9_1':['c9_0'],'t2_3':['t2_2'],'kmax_2':['kmax_1'],'c9_0':[],'result_val_IV_30':['result_13'],'ln_nf_2':[],'ln_nf_3':['ln_nf_2'],'kmax_0':[],'kmax_1':['kmax_0'],'result_val_IV_31':['result_14'],'result_val_IV_32':['result_15'],'result_val_IV_33':['result_val_IV_32'],'result_val_IV_34':['result_val_IV_33'],'g_1':['lng_ser_0','sin_ser_0'],'g_0':['eps_0','c0_2','eps_0','c1_2','eps_0','c2_4','eps_0','c3_2','eps_0','c4_2','eps_0','g5_0'],'t2_0':['x_2'],'g_3':['eps_3','c1_7','eps_3','c2_9','eps_3','c3_7','eps_3','c4_7','eps_3','c5_5','eps_3','g6_0'],'g_2':['g_0','g_1'],'c0_err_0':['c0_3'],'c0_err_1':['c0_err_0'],'Ag_3':['Ag_0','Ag_1'],'Ag_2':['Ag_0','Ag_1'],'Ag_1':['Ag_2','lanczos_7_c_k_IV_0','x_9','k_1'],'Ag_0':[],'stat_psi_1':['x_6','r_0'],'stat_psi_0':['fx_0','result_7'],'stat_psi_2':['stat_psi_0','stat_psi_1'],'result_val_IV_24':['result_10'],'result_val_IV_25':['result_10'],'result_val_IV_26':['result_val_IV_24','result_val_IV_25'],'result_val_IV_27':['result_11'],'result_val_IV_20':['result_val_IV_19'],'result_val_IV_21':['result_9'],'result_val_IV_22':['result_9'],'result_val_IV_23':['result_val_IV_21','result_val_IV_22'],'result_val_IV_28':['result_val_IV_27'],'result_val_IV_29':['result_12'],'result_c_val_IV_4':['result_c_1'],'result_c_val_IV_5':['result_c_1'],'result_c_val_IV_2':['result_c_0'],'result_c_val_IV_3':['result_c_val_IV_1','result_c_val_IV_2'],'result_c_val_IV_0':['result_c_0'],'t1_3':['t1_0','t1_1','t1_2'],'result_c_val_IV_1':['result_c_val_IV_0'],'t1_2':['x_2'],'c8_2':[],'t1_4':['t1_3'],'c8_1':['c8_0'],'c8_0':[],'ser_3':['ser_2'],'ser_0':['ni2_0','ni2_0','c0_0','ni2_0','c1_0','c2_0','ni2_0'],'lng_ser_0':['c0_val_IV_0','eps_0','c1_3','eps_0','c2_5','eps_0','c3_3','eps_0','c4_3','eps_0','c5_3','eps_0','c6_1','eps_0','c7_1'],'ser_1':['ser_0'],'ser_2':['ni2_2','c2_2','ni2_2','c3_0','ni2_2','c4_0','ni2_2','c5_0'],'t1_1':['x_2'],'lng_ser_1':['lng_ser_0'],'t1_0':['x_2'],'v_0':['x_2'],'v_2':['x_2'],'v_1':['x_2'],'v_4':['v_3'],'v_3':['v_0','v_1','v_2'],'lg_z_0':[],'c7_3':[],'result_val_IV_1':['result_val_IV_0'],'eps5_0':['eps_1','eps_1','eps_1','eps_1','eps_1'],'result_val_IV_0':['result_2'],'eps5_1':['eps_2','eps_2','eps_2','eps_2','eps_2'],'c7_0':[],'c7_2':['c7_0','c7_1'],'lg_z_2':['lg_z_1'],'c7_1':['psi_6_val_IV_0'],'lg_z_1':['lg_z_0'],'psi_6_val_IV_0':['psi_6_0'],'psi_6_val_IV_1':['psi_6_val_IV_0'],'e_1':['e_2','y2_0','temp_0','dd_2','cs_c_j_IV_0'],'e_0':[],'e_3':['e_0','e_1'],'e_2':['e_0','e_1'],'e_4':['e_3','y_0','temp_3','dd_3','cs_c_0_IV_0'],'jmax_2':['jmax_1'],'jmax_1':['jmax_0'],'jmax_0':[],'hzeta_err_IV_3':['hzeta_err_IV_2'],'hzeta_err_IV_1':['hzeta_err_IV_0'],'hzeta_err_IV_2':['hzeta_2'],'hzeta_err_IV_0':['hzeta_0'],'lgn_val_IV_0':['lgn_0'],'result_val_IV_3':['result_val_IV_1','result_val_IV_2'],'result_val_IV_2':['result_2'],'result_val_IV_5':['result_2'],'result_val_IV_4':['result_2'],'result_val_IV_7':['result_val_IV_4','result_val_IV_5','result_val_IV_6'],'result_c_val_IV_8':['result_c_val_IV_3','result_c_val_IV_7'],'result_val_IV_6':['result_2'],'result_c_val_IV_6':['result_c_1'],'result_val_IV_9':['result_3'],'result_c_val_IV_7':['result_c_val_IV_4','result_c_val_IV_5','result_c_val_IV_6'],'result_val_IV_8':['result_val_IV_3','result_val_IV_7'],'ln_nf_val_IV_1':['ln_nf_val_IV_0'],'ln_nf_val_IV_0':['ln_nf_0'],'psi_4_0':[],'psi_4_1':['psi_4_0'],'pcp_2':['pcp_0','pcp_1'],'result_err_37':['eMN_0','eab_0'],'c6_1':['psi_5_val_IV_0'],'pcp_3':['pcp_0','pcp_1'],'result_err_38':['result_4'],'c6_0':[],'pcp_4':['pcp_3'],'result_err_39':['result_err_38','eMN_0','eab_0','dy_0','y_2'],'c6_3':[],'pcp_5':['pcp_4'],'c6_2':['c6_0','c6_1'],'ln_nf_val_IV_3':['ln_nf_val_IV_2'],'ln_nf_val_IV_2':['ln_nf_2'],'result_err_40':['result_4'],'result_err_41':['result_err_40','eMN_0','eab_0','dx_0'],'result_err_42':['result_err_41'],'result_err_43':['result_err_33','result_err_36','result_err_42'],'d_0':[],'result_err_44':['result_7'],'result_err_45':['result_err_44','M_3','sum_3'],'d_2':['d_0','d_1'],'pcp_0':['pmax_0','kmax_0','q_0'],'result_err_46':['r_err_IV_0','d_5'],'d_1':['y2_0','d_2','dd_2','cs_c_j_IV_0'],'pcp_1':['pcp_2','kmax_0','q_0','kmax_0','q_0'],'result_err_47':['result_err_45','result_err_46'],'d_4':['y_0','d_3','dd_3','cs_c_0_IV_0'],'d_3':['d_0','d_1'],'max_bits_1':['max_bits_0'],'d_6':['d_5'],'max_bits_0':[],'d_5':['sin_px_0','sin_px_0'],'corr_0':['eps5_0','c0_5','eps_1','c1_5','eps_1','c2_7','eps_1','c3_5','c4_5','eps_1'],'eps_6':['eps_5'],'eps_5':['eps_4'],'eps_4':['x_10','N_4'],'corr_1':['eps5_1','c0_6','eps_2','c1_6','eps_2','c2_8','eps_2','c3_6','c4_6','eps_2'],'return_val_0':['x_10','result_15'],'eps_7':['eps_6'],'t_0':['y_1','y_1'],'return_val_1':['x_10','result_15','sgn_7'],'return_val_2':['N_4','eps_4','result_15','sgn_8'],'result_c_err_IV_1':['result_c_err_IV_0'],'return_val_3':['return_val_2'],'result_c_err_IV_0':['result_c_0'],'t_1':['t_0'],'return_val_4':['return_val_3'],'result_c_err_IV_3':['result_c_err_IV_1','result_c_err_IV_2'],'result_err_26':['result_err_15','result_err_20','result_err_25'],'return_val_5':['return_val_0','return_val_1','return_val_4'],'result_c_err_IV_2':['result_c_0'],'result_err_27':['result_err_10','result_err_26'],'result_c_err_IV_5':['result_c_1'],'result_err_28':['result_val_IV_9'],'result_c_err_IV_4':['result_c_1'],'result_err_29':['s_3','result_val_IV_10'],'result_c_err_IV_7':['result_c_err_IV_4','result_c_err_IV_5','result_c_err_IV_6'],'result_c_err_IV_6':['result_c_1'],'result_c_err_IV_8':['result_c_err_IV_3','result_c_err_IV_7'],'hzeta_val_IV_0':['hzeta_0'],'result_err_30':['jmax_0','ans_6'],'result_err_31':['result_err_28','result_err_29','result_err_30'],'result_err_32':['result_err_31'],'result_err_33':['dy_0','x_3'],'result_err_34':['ex_0','dy_0','y_2','dx_0'],'hzeta_val_IV_3':['hzeta_val_IV_2'],'result_err_35':['result_4'],'hzeta_val_IV_2':['hzeta_2'],'result_err_36':['result_err_35','result_val_IV_13'],'hzeta_val_IV_1':['hzeta_val_IV_0'],'gam_e_0':['g_0','eps_0','eps_0','eps_0','eps_0'],'gam_e_1':['gam_e_0'],'eMN_0':['M_0','N_0'],'psi_5_1':['psi_5_0'],'c5_5':[],'psi_5_0':[],'result_err_15':['result_err_14','result_val_IV_4'],'c5_2':[],'sgn_2':['eps_0'],'result_err_16':['t1_1','x_2','t2_1','t2_1'],'c5_1':['c5_0'],'sgn_1':[],'result_err_17':['result_2'],'eMN_2':['eMN_1'],'sgn_4':['sgn_1','sgn_2','sgn_3'],'c5_4':['c5_2','c5_3'],'result_err_18':['result_err_17','result_c_err_IV_5'],'eMN_1':['eMN_0'],'c5_3':['psi_4_val_IV_0'],'sgn_3':['N_3','eps_0'],'result_err_19':['result_2'],'psi_table_n_IV_0':['n_5'],'c5_0':[],'psi_table_n_IV_1':['psi_table_n_IV_0'],'sgn_9':['sgn_8'],'result_err_20':['result_err_19','result_val_IV_5'],'result_err_21':['t1_2'],'result_err_22':['result_2'],'sgn_6':['eps_3'],'result_err_23':['result_err_22','result_c_err_IV_6'],'c_3':['c_2'],'result_err_24':['result_2'],'sgn_8':[],'c_2':['x_2'],'result_err_25':['result_err_24','result_val_IV_6'],'sgn_7':[],'c_4':['c_3'],'psi_4_val_IV_1':['psi_4_val_IV_0'],'psi_4_val_IV_0':['psi_4_0'],'s_1':['s_0'],'s_0':['x_2'],'s_2':['s_1'],'s_5':['s_4'],'s_4':['z_0'],'result_err_10':['result_err_6','result_err_9'],'result_err_11':['t1_0','x_2','t2_0','t2_0','x_2','t3_0','t3_0'],'result_err_12':['result_2'],'result_err_13':['result_err_12','result_c_err_IV_4'],'result_err_14':['result_2'],'sin_px_1':['sin_px_0'],'sin_px_0':['x_6'],'lanczos_7_c_k_IV_1':['lanczos_7_c_k_IV_0'],'lanczos_7_c_k_IV_2':['lanczos_7_c_k_IV_0'],'lanczos_7_c_k_IV_0':['k_1'],'psi_6_0':[],'c4_7':[],'psi_6_1':['psi_6_0'],'c4_6':[],'c4_3':['psi_3_val_IV_0'],'c4_2':[],'ex_1':['ex_0'],'c4_5':[],'ex_0':['x_3'],'c4_4':['c4_2','c4_3'],'c4_1':['c4_0'],'c4_0':[],'b_4':['b_3'],'b_3':['ly_0','N_0'],'b_5':['b_4'],'psi_1_val_IV_0':['psi_1_0'],'r_0':[],'r_1':['r_0'],'psi_1_val_IV_1':['psi_1_val_IV_0'],'fact_table_n_f_IV_0':['n_6'],'fact_table_n_f_IV_1':['fact_table_n_f_IV_0'],'c3_7':[],'c3_4':['c3_2','c3_3'],'stat_0':['x_10','result_15'],'c3_3':['psi_2_val_IV_0'],'c3_6':[],'stat_2':['stat_0','stat_1'],'c3_5':[],'stat_1':['x_10','result_15'],'c3_0':[],'c3_2':[],'c3_1':['c3_0'],'a_3':['x_3','M_0'],'a_5':['a_4'],'a_4':['a_3'],'result_err_9':['result_err_8','result_val_IV_2'],'result_err_8':['result_2'],'num_0':['eps_1','n1_0','eps_1','n2_0'],'result_err_5':['result_err_4','result_val_IV_0'],'result_err_4':['result_2'],'result_err_7':['result_c_err_IV_2'],'result_err_6':['result_err_5'],'result_err_1':['x_2','s_0','s_0'],'result_err_0':['e_4','cs_c_cs_order_IV_0'],'result_err_3':['result_err_2','result_c_err_IV_0'],'result_err_2':['result_2'],'num_1':['eps_2','n1_1','eps_2','n2_1'],}

#added phi names
phi_names_set = {'dd_2','temp_1','d_2','e_2','cs_c_j_IV_1','dd_3','temp_2','d_3','e_3','cs_c_j_IV_2','result_val_3','result_val_IV_1','result_c_err_IV_1','result_c_val_IV_1','result_err_6','result_val_5','s_1','c_3','result_val_IV_3','result_c_err_IV_3','result_c_val_IV_3','result_err_10','result_val_9','result_val_IV_7','v_3','result_c_err_IV_7','result_c_val_IV_7','result_err_26','t1_3','t2_2','t3_1','c_4','result_c_err_IV_8','result_err_27','result_val_10','s_2','t_1','result_val_IV_8','v_4','result_c_val_IV_8','result_c_2','t1_4','t2_3','t3_2','ans_2','ans_3','scp_2','ans_5','hzeta_c_j_1_IV_1','delta_1','pcp_2','scp_3','ans_6','hzeta_c_j_1_IV_2','delta_2','pcp_3','pmax_1','p1_1','scp_4','p2_1','p3_1','ans_7','delta_3','jmax_1','kmax_1','result_err_31','result_val_14','result_val_IV_11','hzeta_c_j_1_IV_3','pcp_4','pmax_2','p1_2','scp_5','p2_2','p3_2','ans_8','delta_4','max_bits_1','jmax_2','kmax_2','result_err_32','result_val_15','result_val_IV_12','hzeta_c_j_1_IV_4','ln_term0_1','pcp_5','result_val_19','a_4','b_4','sy_1','eMN_1','eab_1','result_err_42','M_1','N_1','a_5','b_5','sy_2','eMN_2','eab_2','result_err_43','ly_1','M_2','N_2','result_val_20','lnr_1','ex_1','result_val_IV_14','result_val_22','result_val_IV_16','ln_nf_1','result_val_23','hzeta_val_IV_1','hzeta_1','result_val_IV_17','stat_hz_1','stat_e_1','stat_nf_1','ln_nf_val_IV_1','ln_nf_err_IV_1','hzeta_err_IV_1','sum_2','sum_3','result_val_27','fx_1','r_1','r_val_IV_1','stat_psi_2','d_6','sin_px_1','sum_4','result_err_47','M_4','r_err_IV_1','result_val_29','result_val_IV_19','ln_nf_3','result_val_30','hzeta_val_IV_3','hzeta_3','result_val_IV_20','stat_hz_3','stat_e_3','stat_nf_3','ln_nf_val_IV_3','ln_nf_err_IV_3','hzeta_err_IV_3','result_val_33','ser_1','result_val_IV_23','ni2_1','result_err_50','c0_1','c1_1','psi_1_table_n_IV_1','c2_1','result_val_36','c3_1','c4_1','ser_3','c5_1','result_val_IV_26','ni2_3','psi_table_n_IV_1','result_err_55','c2_3','result_val_38','result_val_IV_28','result_err_57','fact_table_n_f_IV_1','cs1_1','psi_6_val_IV_1','cs3_1','psi_5_val_IV_1','cs2_1','cs5_1','psi_4_val_IV_1','cs4_1','aeps_1','c0_err_1','psi_0_val_IV_1','psi_0_1','psi_1_1','psi_2_val_IV_1','psi_2_1','c0_val_IV_1','psi_3_1','psi_1_val_IV_1','psi_3_val_IV_1','sgn_4','psi_4_1','psi_5_1','psi_6_1','gam_e_1','lng_err_2','g_2','lng_ser_1','lng_val_2','c0_4','e2_1','sin_ser_1','c1_4','g5_1','c2_6','c3_4','c4_4','c5_4','c6_2','c7_2','lng_val_IV_2','c8_1','c9_1','Ag_2','lanczos_7_c_k_IV_1','Ag_3','lanczos_7_c_k_IV_2','result_val_43','eps_5','sgn_9','return_val_3','result_err_68','N_5','result_val_45','lg_z_1','lg_z_val_IV_1','result_val_IV_33','eps_6','sgn_10','return_val_4','result_err_70','lg_z_err_1','N_6','stat_2','lg_z_2','lg_z_val_IV_2','eps_7','return_val_5','result_err_72','N_7','_as_1','result_val_47','s_5','result_val_IV_34','z_1','sgn_11','lg_z_err_2',}
#--------------------end of progarm-----------

def record_locals(lo, i):
    for name in lo:
        if '_IV' in name:
            continue
        if isinstance(lo[name], numbers.Number) and name in causal_map:
            if name not in global_value_dict:
                columns = causal_map[name].copy()
                columns.insert(0, name)
                global_value_dict[name] = pd.DataFrame(columns=columns)
            new_row = [np.float64(lo[name])]

            for pa in causal_map[name]:
                if isinstance(lo[pa], numbers.Number):
                    new_row.append(np.float64(lo[pa]))
                else:
                    new_row.append(lo[pa])
            global_value_dict[name].loc[i] = new_row

def fluky(good_val, bad_val, p):
        r = random.random()
        if r <= p:
            return bad_val
        else:
            return good_val

bad_dict = {}
global_value_dict = {}
arg1s = np.arange(0.01, 10.01, 0.01)
test_counter = 0


print("Total SSA Variables:", len(causal_map.keys()))
bug = 0 # Ag_1
for arg1 in arg1s:
    bug = fluky(0, 7.4, 0.25)
    bad_outcome = gsl_sf_lngamma(arg1)

    bad_dict[test_counter] = bad_outcome
    test_counter += 1

diff_dict = {index : 0.0 if bad_dict[index] == good_dict[index] else 1.0 for index in bad_dict }


for key in global_value_dict:
    rows = global_value_dict[key].index
    outcome_list = [diff_dict[i] for i in rows]
    global_value_dict[key]['outcome'] = outcome_list


def get_quantiled_tr(W):
    # 10 quantiles from 0.05 to 0.95
    quantile_list = []
    for i in np.arange(0.05, 1.05, 0.1):
        quantile_list.append(W.quantile(i))
    return quantile_list


def predict_causal_risk_list(train_set_X, quantiles, model):

    risk_list = []
    print(train_set_X.columns[0] + " being treated...")
    X_with_quantile = train_set_X.drop(train_set_X.columns[0], axis=1)

    for quantile in quantiles:
        X_with_quantile.insert(loc=0, column=train_set_X.columns[0],
                               value=np.full((len(X_with_quantile), 1), quantile))
        # X_with_quantile[train_set_X.columns[col_index_todrop]] = np.full((len(X_with_quantile), 1), quantile)
        # print(X_with_quantile.describe())
        risk_list.append(model.predict(X_with_quantile).mean())
        X_with_quantile = X_with_quantile.drop(train_set_X.columns[0], axis=1)
    return risk_list



def suspicious_ranking(global_value_dict, model_to_use):

    suspicious_df = pd.DataFrame(columns=['variable_name', 'max_risk_diff', 'quantile1', 'quantile2'])
    for name in global_value_dict:
        if name in phi_names_set:
            continue

        #df cleaning
        #df = global_value_dict[name].select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        df = global_value_dict[name].select_dtypes(include=[np.number]).dropna(axis=1, how='any')
        train_set = df
        #train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        train_set_X = train_set.drop(['outcome'], axis=1)
        train_set_Y = train_set['outcome']
        if model_to_use == 0:
            model = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
        if model_to_use == 1:
            model = Lasso(alpha=0.1)

        
        model.fit(train_set_X, train_set_Y)

        W = df.iloc[:, 0].to_frame()
        quantiles = get_quantiled_tr(W)
        risk_list = predict_causal_risk_list(train_set_X, quantiles, model)
        max_risk = max(risk_list)
        min_risk = min(risk_list)
        row = [df.columns[0], max_risk - min_risk, risk_list.index(max_risk),
               risk_list.index(min_risk)]
        suspicious_df.loc[len(suspicious_df)] = row
    suspicious_df = suspicious_df.sort_values(by='max_risk_diff', ascending=False)
    return filter_phi_rows(suspicious_df, phi_names_set)

def filter_phi_rows(suspicious_df, phi_names_set):
    return suspicious_df[~suspicious_df['variable_name'].isin(phi_names_set)]


# 0-> random forest  1 -> lasso
result = suspicious_ranking(global_value_dict, 0)
pd.set_option("display.precision", 8)
print('*************Target variables in total: ', len(result),'*************')
print(result)