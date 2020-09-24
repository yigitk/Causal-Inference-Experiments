rm(list = ls())
source('C:/Users/yigit/Desktop/Single_Cov.R')
for(dir1 in list.dirs(path = "C:/Users/yigit/Desktop/JiachengExperiments-master/Repeat", full.names = TRUE, recursive = FALSE) ){
  
  
  dir<-paste(dir1,"/Data",sep="")

  n1=1
  datafolder<-1

  n<-1
  print(dir)
  Klosgen<- rep(0, n)
  Ochiai<- rep(0, n)
  F1<- rep(0, n)
  Dstar<-  rep(0, n)
  
  percent_Klosgen<- rep(0, n)
  percent_Ochiai<- rep(0, n)
  percent_F1<- rep(0, n)
  percent_Dstar<- rep(0, n)
  
  
  result<-CoverageScore(dir,datafolder);
  
  Klosgen[datafolder]<- result$Klosgen;
  Ochiai[datafolder]<- result$Ochiai;
  F1[datafolder]<- result$F1;
  Dstar[datafolder]<-  result$Dstar;
  
  percent_Klosgen[datafolder]<- Klosgen[datafolder]/result$n
  percent_Ochiai[datafolder]<- Ochiai[datafolder]/result$n
  percent_F1[datafolder]<- F1[datafolder]/result$n
  percent_Dstar[datafolder]<- Dstar[datafolder]/result$n
  print(result$n)
  
  write(percent_Klosgen,file= paste(dir,"1/coverage.txt",sep = ""), append=TRUE, ncolumns=n, sep=" ")
  write(percent_Ochiai, file=paste(dir,"1/coverage.txt",sep = ""), append=TRUE, ncolumns=n, sep=" ")
  write(percent_F1, file=paste(dir,"1/coverage.txt",sep = ""), append=TRUE, ncolumns=n, sep=" ")
  write(percent_Dstar, file=paste(dir,"1/coverage.txt",sep = ""), append=TRUE, ncolumns=n, sep=" ")
 
}



