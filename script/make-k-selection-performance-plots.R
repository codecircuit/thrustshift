#!/usr/bin/env Rscript

library(ggpubr)
library(tidyverse)
library(scales)

args = commandArgs(trailingOnly=TRUE)


if (length(args)==0) {
	binaries_dir <- "../build/release/"
} else if (length(args)==1) {
	binaries_dir <- args[1]
}

print(binaries_dir)
Ns <- c(2**20)
# for (e in 18:20) {
# 	Ns <- append(Ns, 2**e)
# }

omit_benchmarks <- T
block_dims <- c(256, 512)
num_sh_histograms <- c(1, 2)
num_histograms <- c(68, 2*68, 3*68, 4*68, 8*68)
schemes <- c("vanilla", "atomic", "threadfence", "cub")

extract_csv <- function(content) {
	p0 <- str_locate(content,'"ID","Process ID"')
	if (is.na(p0[1])) {
		print(paste0("ERROR while extracting CSV from content ", content))
	}
	csv_part <- str_replace_all(str_sub(content, p0[1]), "[\r\n ]*$", "")
	df <- read_csv(csv_part, na="N/A")
	return(df)
}

simplify_kernel_name_ <- function(kernel_name) {

	if (str_detect(kernel_name, "sum_subsequent_into")) {
		return("sum_subsequent_into");
	}
	if (str_detect(kernel_name, "bin_values256_atomic")) {
		return("bin_values256_atomic");
	}
	if (str_detect(kernel_name, "bin_values256_threadfence")) {
		return("bin_values256_threadfence");
	}
	if (str_detect(kernel_name, "bin_values256")) {
		return("bin_values256");
	}
	if (str_detect(kernel_name, "fill")) {
		return("fill");
	}
	return("NA");
}
simplify_kernel_name <- Vectorize(simplify_kernel_name_)


df <- NULL
df_curr <- NULL
if (!omit_benchmarks) {
	###########
	# VANILLA #
	###########
	for (N in Ns) {
		for (block_dim in block_dims) {
			for (nsh in num_sh_histograms) {
				for (nh in num_histograms) {
					num_warps <- block_dim %/% 32
					if (num_warps %% num_sh_histograms == 0 && num_sh_histograms <= num_warps) {
						for (scheme in schemes) {
							if (scheme != "cub") {
								# NOTE: parsing failures might occur because the profiler just omits columns in the end. The existing columns are correctly parsed.
								cmd <- paste0("nv-nsight-cu-cli --set full --units base --profile-from-start off  --csv ", binaries_dir, "bench-bin-values256 --N ", N, "  --block_dim  ",block_dim ," --num_sh_histograms ", nsh, " --num_histograms ", nh, " --scheme ", scheme)
								print(cmd)
								out <- paste(system(cmd, intern = TRUE), collapse = '\n')
								df_extr <- extract_csv(out) %>% filter(`Metric Name` %in%  c("Duration", "Memory Throughput"))
								df_curr <- df_curr %>% bind_rows(df_extr %>% mutate(N = N, block_dim = block_dim, num_sh_histograms = nsh, num_histograms = nh, scheme = scheme))
							}
						}
					}
				}
			}
		}
		if ("cub" %in% schemes) {
			scheme <- "cub"
			cmd <- paste0("nv-nsight-cu-cli --set full --units base --profile-from-start off  --csv ", binaries_dir, "bench-bin-values256 --N ", N, " --scheme ", scheme)
			print(cmd)
			out <- paste(system(cmd, intern = TRUE), collapse = '\n')
			df_extr <- extract_csv(out) %>% filter(`Metric Name` %in%  c("Duration", "Memory Throughput"))
			df_curr <- df_curr %>% bind_rows(df_extr %>% mutate(N = N, block_dim = block_dim, num_sh_histograms = nsh, num_histograms = nh, scheme = scheme))
		}
	}

	df_curr <- df_curr %>% mutate(parsed_metric_value = as.double(str_remove_all(`Metric Value`, ","))) %>%
		mutate(kernel_name = simplify_kernel_name(`Kernel Name`)) %>%
		mutate(invocation_id = ID %/% 2)
	df <- df %>% bind_rows(df_curr)
	save.image()
} else {
	load("./.RData")
}

df_tot <- df %>% filter(`Metric Name` == "Duration") %>%
	group_by(N, block_dim, num_sh_histograms, num_histograms, scheme, invocation_id) %>%
	summarise(
		parsed_metric_value = sum(parsed_metric_value)
	) %>%
	group_by(N,num_sh_histograms,num_histograms,block_dim,scheme) %>%
	summarise(duration = mean(parsed_metric_value)) ## mean duration for all histograms

N_curr <- 2**20

df_cub_hline <- NULL
cub_dur <- df_tot %>% filter(scheme == "cub", N == N_curr) %>% select(duration) %>% pull(duration)
for (block_dim in block_dims) {
	for (nsh in num_sh_histograms) {
		for (nh in num_histograms) {
			num_warps <- block_dim %/% 32
			if (num_warps %% num_sh_histograms == 0 && num_sh_histograms <= num_warps) {
				df_curr <- tibble(
					block_dim = block_dim,
					num_sh_histograms = nsh,
					num_histograms = nh,
					duration = cub_dur,
					scheme = "cub"
				)
				df_cub_hline <- df_cub_hline %>% bind_rows(df_curr)
			}
		}
	}
}

df_tot %>%
	filter(N == N_curr, scheme != "cub") %>% 
	mutate(scheme = ifelse(scheme == "vanilla", "bin_values + sum", scheme)) %>%
	ggplot(
		aes(
			x = num_histograms,
			y = duration * 1e-6,
			color = as.factor(scheme),
#			linetype = as.factor(kernel_name)
		)
	) +
	facet_grid(
		rows=vars(num_sh_histograms),
		cols=vars(block_dim),
		labeller = "label_both",
	) +
	geom_hline(aes(yintercept = duration * 1e-6, color = as.factor(scheme)), df_cub_hline) +
	geom_point() +
	geom_line() +
	xlab("num_histograms") +
	ylab("time [ms]") +
	ylim(0, NA) +
	labs(linetype = "kernel", color = "scheme", caption = paste0("N = ", N_curr,", mean time for 1 histogram, 4 histograms created in total"))
ggsave("bin-values-time-averaged.pdf", width=10, height=7)
#ggsave("bin-values-time-averaged.jpg", width=10, height=7)

df %>% filter(`Metric Name` == "Memory Throughput") %>%
	ggplot(
		aes(
			x = N,
			y = parsed_metric_value * 1e-9,
			color = as.factor(ID %/% 2),
			linetype = as.factor(kernel_name)
		)
	) +
	facet_grid(
		rows=vars(num_sh_histograms, block_dim),
		cols=vars(num_histograms),
		labeller = "label_both",
	) +
	geom_point() +
	geom_line() +
	ylab("throughput [GB/s]") +
	xlab("N") +
	labs(linetype = "kernel", color = "invocation") +
	scale_x_continuous(trans = log2_trans(),
	                        breaks = trans_breaks("log2", function(x) 2^x),
	                        labels = trans_format("log2", math_format(2^.x)))

ggsave("bin-values-tp.pdf", width=20, height=20)

save.image()
