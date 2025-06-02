setwd("")
count_dict <- list()
source_dict <- list()


folders <- c("result_data_fold_0", "result_data_fold_1", 
             "result_data_fold_2", "result_data_fold_3")


for (folder in folders) {
  snp_data <- read.csv(file.path(folder, "significant_nodes.csv"), 
                       colClasses = c("character", "character", "numeric"))
  unique_snps <- unique(snp_data$SNP_Name)
  

  for (snp in unique_snps) {

    if (exists(snp, where = count_dict)) {
      count_dict[[snp]] <- count_dict[[snp]] + 1
    } else {
      count_dict[[snp]] <- 1
    }
    

    if (exists(snp, where = source_dict)) {
      source_dict[[snp]] <- c(source_dict[[snp]], folder)
    } else {
      source_dict[[snp]] <- folder
    }
  }
}


result_df <- data.frame(
  SNP = names(count_dict),
  Count = unlist(count_dict),
  Sources = sapply(names(source_dict), function(x) paste(source_dict[[x]], collapse = ", ")),
  Percentage = round(unlist(count_dict)/length(folders)*100, 1),
  stringsAsFactors = FALSE
) 


result_df <- result_df[order(-result_df$Count), ]
row.names(result_df) <- NULL  


head(result_df, 10)