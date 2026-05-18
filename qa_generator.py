def auto_generate_qa_with_chunks(all_data, num_text=15, num_table=10, num_figure=10):
    qa_list = []
    for pnum, items in all_data.items():
        for it in items:
            if it["type"] in ["heading","text"] and len(qa_list) < num_text:
                q = f"请解释第{pnum}页中关于'{it['content'][:30]}'的内容"
                ans = it["content"][:100]
                qa_list.append((q, ans, [pnum], [it["content"]]))
            elif it["type"] == "table" and len(qa_list) < num_text + num_table:
                lines = it["content"].split("\n")
                if len(lines) >= 2:
                    cells = [c.strip() for c in lines[1].split("|") if c.strip()]
                    if len(cells) >= 2:
                        q = f"在第{pnum}页的表格中，{cells[0]} 对应的数值是多少？"
                        ans = cells[1]
                    else:
                        q = f"表格 {pnum} 显示了什么？"
                        ans = lines[1][:100]
                else:
                    q = f"第{pnum}页的表格内容是什么？"
                    ans = it["content"][:100]
                qa_list.append((q, ans, [pnum], [it["content"]]))
            elif it["type"] == "figure" and len(qa_list) < num_text + num_table + num_figure:
                caption = it["content"].replace("[IMAGE_CAPTION]\n","").replace("[CHART_STRUCTURED]\n","")[:100]
                q = f"描述第{pnum}页的图像内容。"
                ans = caption
                qa_list.append((q, ans, [pnum], [it["content"]]))
    return qa_list