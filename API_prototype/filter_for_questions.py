import json
import random
from collections import defaultdict

random.seed(42)

with open('generated_questions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

TARGET_S = {'safety': 25, 'operation': 30, 'troubleshooting': 20, 'maintenance': 15, 'specs': 10}
TARGET_Q = {'factual': 40, 'procedural': 35, 'diagnostic': 25}
TOTAL_TARGET = 100

groups = defaultdict(list)
for q in data:
    groups[(q['source_class'], q['question_class'])].append(q)

avail_S = defaultdict(int)
for q in data:
    avail_S[q['source_class']] += 1

avail_Q = defaultdict(int)
for q in data:
    avail_Q[q['question_class']] += 1

minors = [sc for sc in TARGET_S if avail_S[sc] < TARGET_S[sc]]
majors = [sc for sc in TARGET_S if sc not in minors]


selected = []
used_ids = set()
cur_Q = defaultdict(int)

for sc in minors:
    for qc in TARGET_Q:
        cell_key = (sc, qc)
        if cell_key in groups:
            for item in groups[cell_key]:
                if item['question_id'] not in used_ids:
                    selected.append(item)
                    used_ids.add(item['question_id'])
                    cur_Q[qc] += 1


need_Q = {qc: max(0, TARGET_Q[qc] - cur_Q[qc]) for qc in TARGET_Q}
total_need = sum(need_Q.values())


for qc in TARGET_Q:
    if cur_Q[qc] > TARGET_Q[qc]:
        print(f"  Внимание: {qc} уже превышен ({cur_Q[qc]} > {TARGET_Q[qc]}), не берём из мажоров")
        need_Q[qc] = 0


rem_avail = {sc: {qc: len([x for x in groups[(sc, qc)] if x['question_id'] not in used_ids]) 
                  for qc in TARGET_Q} for sc in majors}

while len(selected) < TOTAL_TARGET:
    unfilled_qc = [qc for qc in TARGET_Q if cur_Q[qc] < TARGET_Q[qc]]
    if not unfilled_qc:
        print(f"  Warning: все question_class заполнены, но всего выбрано {len(selected)} < {TOTAL_TARGET}")
        break
    
    best_choice = None
    best_priority = -1
    
    for qc in unfilled_qc:
        for sc in majors:
            if rem_avail[sc][qc] > 0:
                q_deficit = TARGET_Q[qc] - cur_Q[qc]
                fair_share = (len(selected) - sum(avail_S[m] for m in minors)) / len(majors)
                current_from_major = sum(1 for x in selected if x['source_class'] == sc and sc in majors)
                balance_penalty = current_from_major - fair_share
                
                priority = q_deficit * 10 - balance_penalty
                if priority > best_priority:
                    best_priority = priority
                    best_choice = (sc, qc)
    
    if best_choice is None:
        print(f"  Warning: не осталось доступных записей для добора, останов на {len(selected)}")
        break
    
    sc, qc = best_choice
    # Берём один случайный вопрос из этой ячейки
    pool = [x for x in groups[(sc, qc)] if x['question_id'] not in used_ids]
    if pool:
        chosen = random.choice(pool)
        selected.append(chosen)
        used_ids.add(chosen['question_id'])
        cur_Q[qc] += 1
        rem_avail[sc][qc] -= 1

print(f"Всего выбрано: {len(selected)}")

print("\nРаспределение по question_class:")
for qc in TARGET_Q:
    actual = sum(1 for q in selected if q['question_class'] == qc)
    print(f"  {qc:12s}: выбрано {actual:2d} | таргет {TARGET_Q[qc]:2d} | доступно {avail_Q[qc]:2d}")

print("\nРаспределение по source_class:")
for sc in TARGET_S:
    actual = sum(1 for q in selected if q['source_class'] == sc)
    print(f"  {sc:15s}: выбрано {actual:2d} | таргет {TARGET_S[sc]:2d} | доступно {avail_S[sc]:2d}")


with open('selected_100_questions.json', 'w', encoding='utf-8') as f:
    json.dump(selected, f, ensure_ascii=False, indent=2)
print(f"\nРезультат сохранён в selected_100_questions.json")