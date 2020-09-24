rm(list = ls())
source('C:/Users/yigit/Desktop/CoverageScore.R')
#source('C:/Users/yigit/Desktop/CoverageScore_Multi.R')

dir1<-"C:/Users/yigit/Desktop/TestingWorkspaces/TestEigenDecomposition/Data";
#dir2<-"C:/Users/yigit/Desktop/TestingWorkspaces/TestSimpleRegression/Data";
# dir3<-"C:/Users/zhuofu/workspace/apacheCommonMath3.1.1/TestBlockRealMatrix/Data";
# dir4<-"C:/Users/zhuofu/workspace21/apacheCommonMath3.1.1/TestRotationAndVector3D/Data";
n1=1
datafolder<-1
# n2=2;
# n3=12
# n4=12;

Dirs=c(dir1)
vers=c(n1)



for(index in 1:length(Dirs)){
  dir<-Dirs[index]
  n<-vers[index]
  print(dir)
  Klosgen<- rep(0, n)
  Ochiai<- rep(0, n)
  F1<- rep(0, n)
  Dstar<-  rep(0, n)
  
  percent_Klosgen<- rep(0, n)
  percent_Ochiai<- rep(0, n)
  percent_F1<- rep(0, n)
  percent_Dstar<- rep(0, n)
    
  # for(datafolder in 1:n){
    result<-CoverageScore(dir,datafolder);
    #result<-CoverageScore_Multi(dir,datafolder);
    Klosgen[datafolder]<- result$Klosgen;
    Ochiai[datafolder]<- result$Ochiai;
    F1[datafolder]<- result$F1;
    Dstar[datafolder]<-  result$Dstar;
    
    percent_Klosgen[datafolder]<- Klosgen[datafolder]/result$n
    percent_Ochiai[datafolder]<- Ochiai[datafolder]/result$n
    percent_F1[datafolder]<- F1[datafolder]/result$n
    percent_Dstar[datafolder]<- Dstar[datafolder]/result$n
  print(result$n)
  # }
  write(percent_Klosgen, file="Klosgen_large", append=TRUE, ncolumns=n, sep=" ")
  write(percent_Ochiai, file="Ochiai_large", append=TRUE, ncolumns=n, sep=" ")
  write(percent_F1, file="F1_large", append=TRUE, ncolumns=n, sep=" ")
  write(percent_Dstar, file="Dstar_large", append=TRUE, ncolumns=n, sep=" ")
  file.remove("K_meta.csv")
  file.remove("O_meta.csv")
  file.remove("F_meta.csv")
  file.remove("D_meta.csv")
  file.remove("K_out.csv")
  file.remove("O_out.csv")
  file.remove("F_out.csv")
  file.remove("D_out.csv")
}



