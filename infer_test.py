from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "./models/red-green-dragon-src-7b"
print(f"Loading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Model loaded.")

cuda_code = """
#include <stdio.h>
__global__ void add(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] + b[idx];
}
int main() {
    int a[5] = {1, 2, 3, 4, 5};
    int b[5] = {10, 20, 30, 40, 50};
    int c[5];
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, 5 * sizeof(int));
    cudaMalloc((void**)&d_b, 5 * sizeof(int));
    cudaMalloc((void**)&d_c, 5 * sizeof(int));
    cudaMemcpy(d_a, a, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, 5 * sizeof(int), cudaMemcpyHostToDevice);
    add<<<1, 5>>>(d_a, d_b, d_c);
    cudaMemcpy(c, d_c, 5 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; i++) {
        printf("%d ", c[i]);
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}
"""

prompt = f"Convert the following CUDA code to AMD GPU code:\n```cuda\n{cuda_code}\n```"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Generating HIP code...")
generated_ids = model.generate(**inputs, max_new_tokens=4096)
output = tokenizer.batch_decode(
    [ids[len(inp):] for inp, ids in zip(inputs.input_ids, generated_ids)],
    skip_special_tokens=True,
)[0]

hip_code = output.split("```amd")[-1].split("```")[0]
print("\n=== Raw model output ===")
print(output)
print("\n=== Parsed HIP code ===")
print(hip_code)
