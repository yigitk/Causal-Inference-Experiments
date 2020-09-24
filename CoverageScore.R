CoverageScore <- function (dir,datafolder) {
  source('/Users/yigit/Desktop/sampleOutcome.R')
  y_raw<-read.table(paste(c(dir,as.character(datafolder),"/out.txt"),collapse="")); 
  diff_raw<-read.table(paste(c(dir,as.character(datafolder),"/diff.txt"),collapse=""));
  truth_raw<-read.table(paste(c(dir,as.character(datafolder),"/truth.txt"),collapse=""));
  info<-read.table(paste(c(dir,as.character(datafolder),"/info.txt"),collapse=""))
  info<-as.matrix(info)
  n<-as.numeric(info[1])
  
 # bugid<-as.numeric(info[2])
  
  print(c("n is",n))
  #print(c("bugid is",bugid))
  bugid1<-as.numeric(info[2])
  #bugid2<-as.numeric(info[3])
 
  print(c("bugid1 is",bugid1))
  #print(c("bugid2 is",bugid2))
  
  result = rep(0, n);
  
  result_K=result;
  result_O=result;
  result_F=result;
  result_D=result;
  
  for (i in 1:n) {
    sampleindex<-sampleOutcome(y_raw[,2]);###sample data
    y_raw <- y_raw[sampleindex, ];
    
    y = data.matrix(y_raw[, 2])
    
    
    filename<-paste(c(dir,as.character(datafolder),"/",as.character(i), ".txt"),collapse="")   
    if (!file.exists(filename)){    
      print("not exist");
      print(i)
      next
    }    
    data_raw <- read.table(filename, skip = 1,fill=T)## read variables
    
    
    index0 = which(y < 0.5)
    index1 = which(y > 0.5)
    
    pass=y_raw[index0,];
    fail=y_raw[index1,];
   
    mps=intersect(pass[,1], data_raw[,1]);
    mfs=intersect(fail[,1], data_raw[,1]);
    
    ps=length(mps)
    fs=length(mfs)
    ns=ps+fs;
    p=length(pass[,1]);
    f=length(fail[,1]);
    
    ncf=fs;
    nuf=f-fs;
    ncs=ps;
    
    
    scoreK=sqrt((fs)/(p+f))*((fs/ns)-(f/(p+f)));
    scoreO=fs/sqrt(f*ns);
    scoreF=2/(1/(fs/f)+1/(fs/ns));
    scoreD=ncf*ncf*ncf/(nuf+ncs);
    
    result_K[i]=scoreK;
    result_O[i]=scoreO;
    result_F[i]=scoreF;
    result_D[i]=scoreD;
    
  }
  new_index=length(info)-2
  #new_index=length(info)-3
  Statement<-info[3:new_index]
  #Statement<-info[4:new_index]
  new_bugid1<-gsub("_.*","",info[bugid1+2])
  #new_bugid2<-gsub("_.*","",info[bugid2+3])
  print(new_bugid1)
  
  #print(new_bugid2)
  
  new_data_K=cbind(Statement,result_K)
  new_data_O=cbind(Statement,result_O)
  new_data_F=cbind(Statement,result_F)
  new_data_D=cbind(Statement,result_D)
  
  write.table(new_data_K,file="K_meta.csv",row.names = FALSE,quote=FALSE,col.names =FALSE,append = FALSE,na="0.0",sep=",")
  write.table(new_data_O,file="O_meta.csv",row.names = FALSE,quote=FALSE,col.names =FALSE,append = FALSE,na="0.0",sep=",")
  write.table(new_data_F,file="F_meta.csv",row.names = FALSE,quote=FALSE,col.names =FALSE,append = FALSE,na="0.0",sep=",")
  write.table(new_data_D,file="D_meta.csv",row.names = FALSE,quote=FALSE,col.names =FALSE,append = FALSE,na="0.0",sep=",")
  
  new_result_K<-read.csv(file="C:/Users/yigit/Desktop/K_out.csv",header = T);
  new_result_O<-read.csv(file="C:/Users/yigit/Desktop/O_out.csv",header = T);
  new_result_F<-read.csv(file="C:/Users/yigit/Desktop/F_out.csv",header = T);
  new_result_D<-read.csv(file="C:/Users/yigit/Desktop/D_out.csv",header = T);
  
  indexing<-match(as.character(new_bugid1),new_result_K[,1])
  #indexing2<-match(as.character(new_bugid2),new_result_K[,1])
  # Klosgen<-min(nrow(new_result_K)-rank(new_result_K[,2])[indexing1]+1,nrow(new_result_K)-rank(new_result_K[,2])[indexing2]+1);
  # Ochiai<-min(nrow(new_result_O)-rank(new_result_O[,2])[indexing1]+1,nrow(new_result_O)-rank(new_result_O[,2])[indexing2]+1);
  # F1<-min(nrow(new_result_F)-rank(new_result_F[,2])[indexing1]+1,nrow(new_result_F)-rank(new_result_F[,2])[indexing2]+1);
  # Dstar<-min(nrow(new_result_D)-rank(new_result_D[,2])[indexing1]+1,nrow(new_result_D)-rank(new_result_D[,2])[indexing2]+1);
  print("indexing")
  print(indexing)
  Klosgen<-nrow(new_result_K)-rank(new_result_K[,2])[indexing]+1;
  Ochiai<-nrow(new_result_O)-rank(new_result_O[,2])[indexing]+1;
  F1<-nrow(new_result_F)-rank(new_result_F[,2])[indexing]+1;
  Dstar<-nrow(new_result_D)-rank(new_result_D[,2])[indexing]+1;
  
  
  return(list(Klosgen=Klosgen,Ochiai=Ochiai,F1=F1,Dstar=Dstar,n=nrow(new_result_K),bugid=new_bugid1));
}