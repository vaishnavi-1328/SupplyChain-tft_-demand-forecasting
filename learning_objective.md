I want you to act as my **Senior Supply Chain Data Scientist + Supply Chain Analytics mentor**.

My goal is to become industry-ready for Supply Chain Data Scientist / Supply Chain ML / Decision Science roles by building a realistic portfolio from scratch.

I do NOT want generic projects such as:

* "Build a demand forecasting model"
* "Create an inventory dashboard"
* "Predict sales using XGBoost"
* "Optimize delivery routes"

Instead, I want to build **one realistic end-to-end retail supply-chain simulation** inspired by companies such as Meijer, Walmart, Target, Kroger, Costco, Amazon, and large grocery/retail organizations.

We will use simulated data where necessary, but the business problems, data architecture, decisions, constraints, KPIs, and modeling approaches should resemble what an actual supply-chain data science team would encounter.

## THE SIMULATED COMPANY


Assume it has:

* 300 stores
* 8 distribution centers
* 5,000–10,000 SKUs
* Grocery, FMCG, household, health & beauty, and general merchandise
* Hundreds of suppliers
* Regional suppliers and national suppliers
* Perishable and non-perishable products
* Store-level inventory
* DC inventory
* Purchase orders
* Supplier lead times
* Transportation between suppliers → DCs → stores
* Promotions
* Holidays
* Weather effects
* Product substitutions
* Stockouts
* Returns
* Spoilage/waste
* Damaged products
* Supplier disruptions
* Transportation delays

I want this to behave like a **digital laboratory for supply-chain decision making**, not merely a dataset for ML.

---

# MY LEARNING OBJECTIVE

By completing this system, I want to understand deeply:

1. Demand forecasting - done
2. Time-series modeling - done
3. Intermittent demand  - done : Croston's method, syntetos boylan approximation
4. Inventory theory - 
5. Safety stock
6. Reorder points
7. Service levels
8. EOQ - (2DS/H)^0.5
9. Multi-echelon inventory
10. Supplier lead-time modeling
11. Supplier reliability
12. Purchase-order analytics
13. Inventory allocation
14. Demand sensing
15. Promotion effects
16. Stockout detection
17. Lost-sales estimation
18. Substitution behavior
19. Forecast uncertainty
20. Probabilistic forecasting
21. Supply disruption modeling
22. Transportation analytics
23. Route/dispatch decisions
24. Distribution-center capacity
25. Warehouse bottlenecks
26. SKU segmentation
27. ABC/XYZ analysis
28. Network optimization
29. Optimization under constraints
30. Simulation
31. Causal reasoning
32. Anomaly detection
33. Scenario analysis
34. Cost-to-serve
35. Working-capital optimization
36. Service-level optimization
37. Explainable ML
38. Production data pipelines
39. Supply-chain data architecture
40. Business communication and decision-making

Do not try to teach everything simultaneously.

Build the knowledge progressively.

---

# THE CORE PROJECT

Design a sequence of interconnected projects around this supply chain.

The projects should form a progression:

### PROJECT 1 — Supply Chain Control Tower

Before ML, help me understand the system.

Build simulated tables for:

* products
* stores
* distribution_centers
* suppliers
* customers
* calendar
* inventory_snapshots
* sales_transactions
* purchase_orders
* purchase_order_lines
* shipments
* shipment_events
* promotions
* returns
* spoilage
* weather
* product_supplier relationships

Teach me the relationships between these tables.

Create a realistic warehouse/star-schema architecture.

Explain:

OLTP → ingestion → data warehouse → feature layer → ML → optimization → decision layer.

I should be able to explain the architecture in a technical interview.

---

# PROJECT 2 — "Why Did We Run Out?"

Build a **Stockout Investigation Engine**.

Instead of simply predicting sales, the system should answer:

> "Why was SKU X unavailable at Store Y on March 17?"

Potential root causes:

* unexpected demand spike
* inaccurate forecast
* supplier delay
* DC shortage
* store ordering behavior
* insufficient safety stock
* transportation delay
* promotion
* weather
* incorrect inventory records
* allocation problem
* supplier fill-rate failure

Build a rule-based baseline first.

Then introduce ML/anomaly detection.

The output should explain the likely root cause.

This project should teach me how supply-chain data scientists investigate operational failures.

---

# PROJECT 3 — "What Should We Order?"

Build a **Replenishment Decision Engine**.

For every:

SKU × Store × Day

calculate:

* forecast demand
* lead-time demand
* demand variability
* lead-time variability
* safety stock
* reorder point
* inventory position
* days of supply
* expected stockout risk
* recommended order quantity

Compare:

1. naive reorder policy
2. EOQ
3. reorder point + safety stock
4. service-level-based policy
5. ML-enhanced replenishment policy

Do NOT optimize only RMSE.

Evaluate:

* fill rate
* stockout rate
* inventory holding cost
* ordering cost
* waste
* working capital
* service level

Teach me why the "best ML model" may produce a worse supply-chain decision.

---

# PROJECT 4 — "Forecast the Future, But Admit Uncertainty"

Build a serious demand forecasting system.

Use:

* baseline moving average
* seasonal naive
* exponential smoothing
* ARIMA/SARIMA
* gradient boosting
* XGBoost/LightGBM
* hierarchical forecasting
* probabilistic forecasting

Features should include:

* historical demand
* day of week
* month
* holidays
* promotions
* price
* weather
* store characteristics
* product characteristics
* regional trends

Do not stop at point predictions.

Produce:

P10 / P50 / P90 forecasts.

Teach me:

> "A forecast of 100 units is not enough for an inventory decision."

Connect forecast uncertainty directly to safety stock.

---

# PROJECT 5 — "The Promotion Trap"

Simulate promotions.

For example:

SKU A normally sells:

100 units/week

During promotion:

180 units/week

But the promotion may cause:

* cannibalization of another SKU
* post-promotion demand dip
* forward buying
* substitution
* store-level differences

Build a system that estimates:

* incremental demand
* baseline demand
* promotion lift
* cannibalization
* post-promotion effects

Teach me the difference between:

correlation vs causal effect.

Introduce causal inference only after the baseline analysis is working.

---

# PROJECT 6 — "Which Supplier Is Actually Reliable?"

Build a Supplier Risk & Reliability Engine.

For every supplier calculate:

* average lead time
* lead-time variance
* fill rate
* OTIF
* cancellation rate
* defect rate
* price volatility
* historical disruptions
* PO confirmation behavior

Then predict:

> "What is the probability that this supplier will deliver this PO late?"

Build:

* baseline statistical model
* classification model
* explainability layer

Then connect supplier risk to inventory policy.

For example:

high-risk supplier → larger safety stock

reliable supplier → lower safety stock

Show me how ML changes an actual procurement decision.

---

# PROJECT 7 — "Where Should Inventory Sit?"

Build a simplified **multi-echelon inventory optimization** problem.

We have:

Supplier → DC → Store

Inventory can exist at different levels.

Teach me:

* echelon inventory
* safety stock positioning
* pooling
* lead time
* service levels
* inventory centralization

Create scenarios where:

Scenario A:
Every store holds large safety stock.

Scenario B:
DC holds more inventory.

Scenario C:
Inventory is dynamically allocated.

Compare:

* total inventory
* service level
* transportation cost
* stockouts
* working capital

This should introduce optimization thinking.

---

# PROJECT 8 — "The DC Is Becoming the Bottleneck"

Build a Distribution Center Capacity Simulator.

Simulate:

* inbound shipments
* outbound store orders
* dock capacity
* picking capacity
* labor availability
* truck arrivals
* SKU velocity
* warehouse congestion

Detect:

> "This DC will exceed capacity in the next 7 days."

Build a forecasting/anomaly detection component.

Then create a decision layer:

* shift shipments
* change replenishment schedule
* prioritize SKUs
* reroute stores
* adjust labor
* delay low-priority orders

This project should teach me that supply-chain DS is not just forecasting.

---

# PROJECT 9 — "A Truck Is Late. Now What?"

Build a transportation disruption simulator.

Example:

Supplier → DC shipment delayed by 2 days.

The system should determine:

* which SKUs are affected
* which stores are affected
* expected stockouts
* revenue at risk
* service-level impact
* alternative inventory sources
* possible transfer decisions

Then build a scenario engine:

"What happens if the shipment is delayed 1/2/3/5 days?"

This should teach me **scenario analysis and supply-chain resilience**.

---

# PROJECT 10 — "Which Store Should Get the Last 100 Units?"

Build an **Inventory Allocation Engine**.

Suppose a DC has:

100 units available.

But stores collectively need:

250.

Do not simply allocate proportionally.

Create an optimization problem considering:

* expected demand
* current inventory
* forecast
* store importance
* service-level target
* lost-sales risk
* perishability
* distance
* transportation cost
* promotion
* urgency

Then formulate the optimization mathematically.

Start with heuristics.

Then use linear/integer optimization.

Explain every constraint.

---

# PROJECT 11 — "Where Is Money Being Lost?"

Build a Supply Chain Profitability Engine.

For every:

SKU × Store

estimate:

Revenue

− product cost

− transportation

− holding cost

− ordering cost

− spoilage

− markdowns

− stockout/lost-sales cost

= contribution margin

Now identify:

* profitable SKUs
* inventory-heavy SKUs
* low-margin high-volume SKUs
* expensive-to-serve stores
* waste-heavy products
* suppliers causing hidden costs

Teach me why supply-chain optimization is ultimately a **financial optimization problem**.

---

# PROJECT 12 — THE FINAL SYSTEM

Combine everything into a:

# SUPPLY CHAIN DECISION INTELLIGENCE PLATFORM

The platform should answer questions such as:

### Inventory

"Which SKUs are likely to stock out next week?"

### Procurement

"Which suppliers are becoming risky?"

### Forecasting

"Where are forecasts least reliable?"

### Transportation

"Which shipments put revenue at risk?"

### Distribution

"Which DCs are approaching capacity?"

### Allocation

"Where should limited inventory go?"

### Finance

"Where are we tying up too much working capital?"

### What-if analysis

"What happens if supplier X is delayed 3 days?"

### Executive question

> "If I have $2 million of additional inventory budget, where should I put it?"

The system should be capable of producing a recommendation and explaining the reasoning.

---

# TECHNICAL STACK

Prefer:

Python
Pandas
NumPy
SQL
PostgreSQL
PySpark
Microsoft Fabric concepts
Azure Data Factory concepts
Power BI
scikit-learn
XGBoost/LightGBM
statsmodels
OR-Tools / optimization library
MLflow
FastAPI
Streamlit

Do not force every technology into every project.

Teach me why each technology belongs where it does.

---

# DATA GENERATION

Because real company data is unavailable, create realistic simulated data.

But do NOT make the data random noise.

Build relationships and business mechanisms.

For example:

Demand should depend on:

seasonality
day of week
promotions
price
weather
store characteristics
product category
holidays

Supplier lead time should depend on:

supplier
route
distance
season
disruption probability

Stockouts should emerge naturally from:

demand
forecast error
inventory policy
lead time
supplier reliability

Promotions should create:

lift
cannibalization
post-promotion effects

Weather should affect appropriate categories.

Perishables should behave differently from durable products.

The simulated data must contain **realistic causal relationships** so that the models have something meaningful to learn.

---

# HOW YOU SHOULD TEACH ME

Do NOT dump the entire project at once.

Teach me like a senior engineer mentoring a junior data scientist.

For each project:

1. Explain the business problem.
2. Explain who in the company cares about it.
3. Explain the decision being made.
4. Explain the data required.
5. Design the schema.
6. Generate the simulated data.
7. Explore the data.
8. Establish a naive baseline.
9. Explain the mathematics.
10. Build the model.
11. Evaluate it using business metrics.
12. Connect the model to a decision.
13. Perform what-if analysis.
14. Build a small production-like pipeline.
15. Build a visualization/dashboard.
16. Explain how this would work in a real company.
17. Give me interview questions.
18. Give me a senior-level system-design question.
19. Give me one failure scenario to debug.

---

# IMPORTANT RULE

Never let me hide behind a model.

Whenever I build an ML model, force me to answer:

> What decision does this prediction change?

> What happens if the prediction is wrong?

> What is the cost of being wrong?

> What operational constraint does the business have?

> What business metric improved?

> Would the company actually deploy this?

---

# PROJECT DEPTH

For every project, teach me at three levels:

### LEVEL 1 — Analyst

SQL
EDA
KPIs
business reasoning

### LEVEL 2 — Data Scientist

statistics
ML
forecasting
uncertainty
experimentation

### LEVEL 3 — Supply Chain Decision Scientist

optimization
simulation
constraints
tradeoffs
cost functions
scenario analysis
decision systems

I want to progress from Level 1 → Level 2 → Level 3.

---

# PORTFOLIO STANDARD

Every completed project should produce:

* GitHub repository
* README
* architecture diagram
* data dictionary
* SQL transformations
* Python pipeline
* model
* evaluation
* business metrics
* dashboard
* scenario analysis
* technical documentation
* business case
* limitations
* future improvements

The project should look like something a **real supply-chain analytics/DS team could have built**, not a Kaggle notebook.

---

# START HERE

Do NOT start by giving me code.

First create the complete learning roadmap.

Show me:

Project number
Business problem
Supply-chain concept learned
Data required
Analytics method
ML method
Optimization method
Decision produced
Industry skill gained
Difficulty
Dependencies on previous projects

Then tell me:

**Which project should I build first and why?**

After that, begin teaching me Project 1 step-by-step.

Do not move to the next project until I understand the current one.
