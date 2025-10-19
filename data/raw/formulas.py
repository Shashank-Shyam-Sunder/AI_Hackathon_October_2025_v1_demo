#Calculation of merged tokens#

def merge_token_savings(io_tokens_in_default, mode, io_addon_merge_ratio, N, alpha=0.25, s=5):
    T_sep = N * io_tokens_in_default
    T_merge_raw = alpha * io_tokens_in_default + N * ((1 - alpha) * io_tokens_in_default + s)
    if in ("separate_only", "fixed"):
        return 0
    elif mode == "proportional":
        T_merge = T_merge_raw * (1 + io_addon_merge_ratio)
        return max(0, T_sep - T_merge)
    else:
        # Unknown mode -> be safe: no savings
        return 0

    """
    s = separator tokens (assumed)
    alpha = shared tokens (assumed)
    N = number of common tasks in one category (internal label - calculated)

    T_sep = input tokens for N tasks
    T_merge_raw = total tokens per grouped category call
    T_merge = total billed token after adjustment
    """ 

#Self-Hosting price LLM

####
if infra_profile.hosting_type == "cloud":
    base_price = gpu_hourly_rate * gpu_count_default * 24 * 30
    final_hosting_price = base_price * availability_multiplier * monitoring_multiplier * compliance_multiplier

availability_multiplier = 1.1
monitoring_multiplier = 1.05
compliance_multiplier = 1.0 


#Qdrant uses compute + storage pricing
compute_cost = 0.10 * 'estimated_qps' * 720  # $0.10 per vCPU-hour
storage_cost = 'total_storage_gb' * 0.15  # $0.15/GB/month

estimated_qps -> queries per second -> comes from queries per month

total_storage_gb = (vector_storage_gb + metadata_storage_gb) * index_overhead

    total_vectors = total_chunks -> document_size/ chunk_size
        
        chunk_size -> from catalogue

        vector_storage_gb = (total_vectors * bytes_per_vector) / (1024*3)
        metadata_storage_gb = vector_storage_gb * 0.15
        index_overhead = 1.4
        
        
#Chunk sizes
Embeddings / indexing (RAG)
num_chunks = (corpus_gb × bytes_per_gb) / (avg_chunk_tokens × bytes_per_token)


bytes_per_gb = 1_000_000_000
Avg_chunk_tokens == granulatity
Bytes_per_token = 5

embed_tokens = num_chunks × avg_chunk_tokens


embedding_cost = embed_tokens × embed_price_per_token(embed_model) × refresh_multiplier(refresh_rate)

refresh_multiplier = 1 + (refresh_rate_fraction * refresh_frequency_per_year)

Num_chunks (client label: granularity of embedding?) = {“less_granular”:150, “standart”: 350, “more_granular”: 1500}



   

#NVIDIA NIM Guardrails


class NvidiaNIMCostCalculator:
    def init(self):
        self.pricing = {
            'input_tokens_per_million': 1.50,      # $1.50 per million input tokens
            'output_tokens_per_million': 1.50,     # $1.50 per million output tokens
            'min_throughput_tier': 50,             # Minimum monthly commitment
            'api_call_base_fee': 0.0001,           # Per-request base fee
        }

    def calculate_nim_costs(self, monthly_requests, avg_input_tokens, avg_output_tokens):
        """Calculate NVIDIA NIM Guardrails costs - Single Formula"""


        effective_cost = max(
            (monthly_requests * avg_input_tokens / 1_000_000) * self.pricing['input_tokens_per_million'] +
            (monthly_requests * avg_output_tokens / 1_000_000) * self.pricing['output_tokens_per_million'] +
            (monthly_requests * self.pricing['api_call_base_fee']),
            self.pricing['min_throughput_tier']
        )

        return effective_cost 
        }
   
#MATH FORMULA   
        
NVIDIA_NIM_Monthly_Cost = max(
    R × [(I + O) × 0.0000015 + 0.0001],
    50
)

Where:
R = Monthly Requests
I = Average Input Tokens per Request  -> io_tokens_in_default
O = Average Output Tokens per Request -> io_tokens_out_default 

NVIDIA_NIM_Monthly_Cost = rpm * [(io_tokens_in_default + io_tokens_out_default) * 0.0000015 + 0.0001], 50
