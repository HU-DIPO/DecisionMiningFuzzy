<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">
<html>
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
  <meta http-equiv="Content-Style-Type" content="text/css">
  <title></title>
  <meta name="Generator" content="Cocoa HTML Writer">
  <meta name="CocoaVersion" content="2485.2">
  <style type="text/css">
    p.p1 {margin: 0.0px 0.0px 0.0px 0.0px; font: 12.0px Helvetica}
    p.p2 {margin: 0.0px 0.0px 0.0px 0.0px; font: 12.0px Helvetica; min-height: 14.0px}
  </style>
</head>
<body>
<p class="p1">import csv</p>
<p class="p1">import random</p>
<p class="p2"><br></p>
<p class="p1">def determine_catheterization(consent, clinical_indications, risk_factor, patient_discomfort):</p>
<p class="p1"><span class="Apple-converted-space">    </span># Based on the provided decision table</p>
<p class="p1"><span class="Apple-converted-space">    </span>if not consent:</p>
<p class="p1"><span class="Apple-converted-space">        </span>return False</p>
<p class="p1"><span class="Apple-converted-space">    </span>if not clinical_indications:</p>
<p class="p1"><span class="Apple-converted-space">        </span>return False</p>
<p class="p1"><span class="Apple-converted-space">    </span>if risk_factor == "high" and patient_discomfort == "medium":</p>
<p class="p1"><span class="Apple-converted-space">        </span>return True</p>
<p class="p1"><span class="Apple-converted-space">    </span>if risk_factor == "high" and patient_discomfort == "high":</p>
<p class="p1"><span class="Apple-converted-space">        </span>return True</p>
<p class="p1"><span class="Apple-converted-space">    </span>if clinical_indications and risk_factor == "low":</p>
<p class="p1"><span class="Apple-converted-space">        </span>return True</p>
<p class="p1"><span class="Apple-converted-space">    </span>if clinical_indications and risk_factor == "medium":</p>
<p class="p1"><span class="Apple-converted-space">        </span>return True</p>
<p class="p2"><br></p>
<p class="p1"><span class="Apple-converted-space">    </span>return False</p>
<p class="p2"><br></p>
<p class="p1">def generate_row(row_num):</p>
<p class="p1"><span class="Apple-converted-space">    </span>consent = random.choice([True, False])</p>
<p class="p1"><span class="Apple-converted-space">    </span>clinical_indications = random.choice([True, False])</p>
<p class="p1"><span class="Apple-converted-space">    </span>risk_factor = random.choice(["low", "medium", "high"])</p>
<p class="p1"><span class="Apple-converted-space">    </span>patient_discomfort = random.choice(["low", "medium", "high"])</p>
<p class="p1"><span class="Apple-converted-space">    </span>determine_cath = determine_catheterization(consent, clinical_indications, risk_factor, patient_discomfort)</p>
<p class="p2"><span class="Apple-converted-space">    </span></p>
<p class="p1"><span class="Apple-converted-space">    </span>return {</p>
<p class="p1"><span class="Apple-converted-space">        </span>"When": row_num,</p>
<p class="p1"><span class="Apple-converted-space">        </span>"Consent": consent,</p>
<p class="p1"><span class="Apple-converted-space">        </span>"Clinical Indications": clinical_indications,</p>
<p class="p1"><span class="Apple-converted-space">        </span>"Risk Factor": risk_factor,</p>
<p class="p1"><span class="Apple-converted-space">        </span>"Patient Discomfort": patient_discomfort,</p>
<p class="p1"><span class="Apple-converted-space">        </span>"Then": determine_cath,</p>
<p class="p1"><span class="Apple-converted-space">        </span>"Annotations": ""</p>
<p class="p1"><span class="Apple-converted-space">    </span>}</p>
<p class="p2"><br></p>
<p class="p1">def main():</p>
<p class="p1"><span class="Apple-converted-space">    </span>rows = []</p>
<p class="p1"><span class="Apple-converted-space">    </span>headers = ["When", "Consent", "Clinical Indications", "Risk Factor", "Patient Discomfort", "Then", "Annotations"]</p>
<p class="p1"><span class="Apple-converted-space">    </span>for i in range(1000):</p>
<p class="p1"><span class="Apple-converted-space">        </span>rows.append(generate_row(i+1))</p>
<p class="p2"><br></p>
<p class="p1"><span class="Apple-converted-space">    </span>with open("decision_table.csv", "w", newline='') as csvfile:</p>
<p class="p1"><span class="Apple-converted-space">        </span>writer = csv.DictWriter(csvfile, fieldnames=headers)</p>
<p class="p1"><span class="Apple-converted-space">        </span>writer.writeheader()</p>
<p class="p1"><span class="Apple-converted-space">        </span>for row in rows:</p>
<p class="p1"><span class="Apple-converted-space">            </span>writer.writerow(row)</p>
<p class="p2"><br></p>
<p class="p1">if __name__ == "__main__":</p>
<p class="p1"><span class="Apple-converted-space">    </span>main()</p>
</body>
</html>
