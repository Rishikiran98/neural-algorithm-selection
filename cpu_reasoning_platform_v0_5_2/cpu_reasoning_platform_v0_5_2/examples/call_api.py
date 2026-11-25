from cpu_platform.client import CPUReasoningClient

client = CPUReasoningClient()
for question in ["calculate: 12*(7+3)", "Maya has 12 apples and buys 8, then shares equally among 5 children. How many each?"]:
    result = client.reason(question)
    print(question)
    print("answer:", result.get("answer"))
    print("route:", result.get("route"))
    print("arithmetic_verified:", result.get("arithmetic_verified"))
    print("semantically_verified:", result.get("semantically_verified"))
    print("status:", result.get("status", result.get("error")))
    print()
