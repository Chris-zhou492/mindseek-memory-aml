"""Add an audited view while preserving whole passages within shared limits."""


def pack(packet, evidence, top_k, budget):
    output, remaining, merged = [dict(packet)], budget - len(packet["content"]), False
    for item in evidence:
        label = f"\n\n[Original retrieved evidence {item['id']}]\n"
        if not merged and len(item["content"]) + len(label) <= remaining:
            output[0]["content"] += label + item["content"]
            output[0]["group_member_ids"] = list(
                dict.fromkeys([*output[0].get("group_member_ids", []), item["id"], *item.get("group_member_ids", [])])
            )
            output[0]["record_sources"] = [*output[0].get("record_sources", []), *item.get("record_sources", [])]
            output[0]["context_message_ids"] = list(
                dict.fromkeys([*output[0].get("context_message_ids", []), *item.get("context_message_ids", [])])
            )
            remaining -= len(item["content"]) + len(label)
            merged = True
        elif len(output) < top_k and len(item["content"]) <= remaining:
            output.append(item)
            remaining -= len(item["content"])
        else:
            output[0].setdefault("budget_omitted_ids", []).append(item["id"])
    return output
