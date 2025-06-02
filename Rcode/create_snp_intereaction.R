setwd("your_data_path")

# 加载必要的包
library(tidyverse)
library(igraph)
library(ggraph)
library(viridis)
library(officer)

# 加载数据 ----------------------------------------------------------------
nodes_df <- read_csv("significant_nodes.csv",
                     col_types = cols(Feature_ID = "c", SNP_Name = "c", Interaction_Count = "d"))

edges_df <- read_csv("significant_pairs.csv",
                     col_types = cols(Feature_A = "c", Feature_B = "c", SNP_A = "c", SNP_B = "c", Weight = "d"))


#基因注释------------------------------------------------------------------
# 加载必要的包
library(GenomicRanges)
library(ChIPseeker)
library(TxDb.Hsapiens.UCSC.hg19.knownGene)  # GRCh37对应hg19
library(org.Hs.eg.db)

# 解析SNP位置信息（假设nodes_df已加载）
nodes_df <- nodes_df %>%
  separate(
    SNP_Name,
    into = c("chr", "pos", "ref", "alt"),
    sep = ":",
    remove = FALSE,
    convert = TRUE  # 自动转换pos为数值
  )

# 创建GRanges对象（GRCh37坐标）
# 确保GRanges对象包含SNP_Name元数据
snp_gr <- GRanges(
  seqnames = paste0("chr", nodes_df$chr),
  ranges = IRanges(start = nodes_df$pos, end = nodes_df$pos),
  SNP_Name = nodes_df$SNP_Name  # 关键：添加SNP标识符
)

# 检查元数据列
mcols(snp_gr)$SNP_Name  # 应输出所有SNP名称

# 执行基因注释（包含启动子区域）
peakAnno <- annotatePeak(
  snp_gr,
  tssRegion = c(-3000, 3000),  # 基因TSS上下游3kb视为关联
  TxDb = TxDb.Hsapiens.UCSC.hg19.knownGene,
  annoDb = "org.Hs.eg.db"
)

library(dplyr)
# 提取注释结果到数据框
anno_df <- as.data.frame(peakAnno) %>%
  dplyr::select(SNP_Name = SNP_Name,  # 假设原数据中保留SNP标识
         geneId = geneId,
         SYMBOL = SYMBOL,
         distanceToTSS = distanceToTSS) %>%
  distinct()  # 去重

# 处理多基因注释（一个SNP可能关联多个基因）
anno_clean <- anno_df %>%
  group_by(SNP_Name) %>%
  summarise(
    Gene = ifelse(
      n() > 1,
      paste(SYMBOL, collapse = ","),  # 多基因用逗号分隔
      SYMBOL
    )
  ) %>%
  ungroup()

nodes_df <- nodes_df %>%
  left_join(anno_clean, by = "SNP_Name") %>%
  mutate(
    Gene = replace_na(Gene, "NoGene"),  # 无基因的标记为NoGene
    Label = paste0(SNP_Name, "\n", Gene)  # 创建组合标签
  )

# 创建全量图对象 ----------------------------------------------------------
library(igraph)

snp_graph <- graph_from_data_frame(
  d = edges_df %>% dplyr::select(from = Feature_A, to = Feature_B, weight = Weight),
  vertices = nodes_df %>%
    dplyr::select(
      name = Feature_ID,
      SNP_Name,
      size = Interaction_Count,
      Label  # 包含组合标签
    ),
  directed = FALSE
)

# 可视化设置 --------------------------------------------------------------
set.seed(1234)

#cairo_ps("snp_interaction.eps", width = 10, height = 7, fallback_resolution = 600)

p <- ggraph(snp_graph, layout = "circle") +
  # 边设置
  geom_edge_link(
    aes(edge_width = weight, edge_color = weight),
    alpha = 0.5,
    edge_width = 0.8,
    show.legend = TRUE
  ) +
  # 节点设置
  geom_node_point(
    aes(size = size),
    color = "#377EB8",     # 深蓝色边框
    fill = "#A6CEE3",      # 浅蓝色填充
    alpha = 0.9,
    stroke = 0.8,
    shape = 21
  ) +
  # 标签设置
  geom_node_text(
    aes(label = Label),
    color = "#2F4F4F",
    size = 2.8,           # 减小字号（原3.2）
    lineheight = 0.7,     # 减小行间距
    repel = TRUE,
    max.overlaps = 50,    # 减少允许重叠数（原40）
    family = "sans",
    fontface = "plain",
    box.padding = 0.5,    # 增大标签间距（原0.5）
    segment.size = 0.3,   # 添加引导线
    segment.alpha = 0.5,  # 引导线透明度
    min.segment.length = 0.1 # 最小引导线长度
  )+
  # 边颜色比例尺
  scale_edge_color_gradientn(
    name = "Edge Weight",
    colours = c("white", "#B2DF8A", "#66C2A5", "#1D91C0"),
    values = scales::rescale(c(0, 0.4, 0.7, 1))  # 手动弯曲颜色变化曲线
  ) +
  # 边宽比例尺
  scale_edge_width(
    range = c(0.3, 2.2),
    guide = guide_legend(override.aes = list(edge_color = "#666666"))
  ) +
  # 节点大小比例尺
  scale_size(
    name = "Interaction Count",
    range = c(2, 9),
    breaks = seq(2, 10, 2)
  ) +
  # 主题
  theme_void() +
  theme(
    legend.position = "right",
    plot.background = element_rect(fill = "white", color = NA),
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 8),
    legend.key.height = unit(0.8, "lines"),
    text = element_text(family = "sans")  # 统一字体
  ) +
  labs(title = NULL)

# 创建新PPT文档
ppt <- read_pptx()

# 添加空白幻灯片
ppt <- add_slide(ppt, layout = "Title and Content", master = "Office Theme")

# 插入图形对象（矢量格式，可编辑）
ppt <- ph_with(ppt, 
               dml(ggobj = p),  # 使用rvg转换ggplot对象
               location = ph_location_fullsize(),  # 全屏显示
               bg = "transparent")

# 保存PPTX文件
print(ppt, target = "snp_interaction.pptx")
#dev.off()

#ggsave("snp_interaction.svg", 
 #      plot = p, 
  #     device = svg, 
   #    width = 10, 
    #   height = 7, 
     #  units = "in",
      # dpi = 300)

# 保存
#ggsave("Figure1.pdf",
 #      width = 180 / 25.4,   # 毫米转英寸（180mm=7.08inches）
  #     height = 120 / 25.4,  # 毫米转英寸（120mm=4.72inches）
   #    device = grDevices::pdf,  # 使用基础PDF设备
    #   colormodel = "cmyk",  # 印刷色域
     #  limitsize = FALSE,     # 解除尺寸限制
      # useDingbats = FALSE)   # 优化矢量线条
