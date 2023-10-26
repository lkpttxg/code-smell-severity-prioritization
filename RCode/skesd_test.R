library(ScottKnottESD)

# In the processed folder from the previous step, there is a folder containing the evaluation metric values.
finalpath = "your path"
# Folder for storing the SKESD results.
skresultpath = "your path"

# Iterate through all CSV files in the 'finalpath' directory and perform the SK_ESD operation.

dirs_list <- list.dirs(finalpath)
#print(dirs_list)
for (i in 1:length(dirs_list)) {
  # Get the names of the files in each folder.
  file_names <- list.files(dirs_list[i])
  
  for (j in 1:length(file_names)) {
    
    if (grepl(pattern = ".csv$", x = file_names[j])) {
      
      path = paste(dirs_list[i],sep = "/",file_names[j])
      #print(path)
      # generate csv
      csv<-read.csv(path, encoding = "UTF-8")
      #  sk_esd
      sk <- 0
      #print(class(sk))
      try(sk<-sk_esd(csv))
      # write to new file
      parent_path =  substr(dirs_list[i],nchar(finalpath)+1, nchar(dirs_list[i]))
      #print(parent_path)
      resultpath = paste(skresultpath, sep = "", parent_path )
      if (!file.exists(resultpath)){
        dir.create(resultpath, recursive = TRUE)
      }
      resultpath = paste(resultpath, sep="/", file_names[j])
      resultpath = paste(resultpath, sep="", ".txt")
      if ((!class(sk) == "numeric")[1]) {
        print(resultpath)
        write.table(sk[["groups"]], resultpath)
      }
      else {
        print(resultpath)
      }
    }
  }
}




