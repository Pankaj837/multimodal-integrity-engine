"""Text content of the synthetic evaluation dataset.

All companies, teams and figures are fictional ("Kestrel Foods Ltd." does not
exist). External-copy passages are quoted from pinned, pre-ChatGPT Wikipedia
revisions (CC BY-SA 4.0) stored in reference_corpus/.

Provenance (stated honestly; see dataset_manifest.json and EVALUATION.md):
every text below was drafted with AI assistance during this
assessment. Files labelled ``ai_generated`` are unedited "default-style" LLM
output produced from a plain instruction. Files labelled ``human_style`` were
written to emulate student writing (uneven sentence length, first person,
informal asides, local detail). Because their true provenance is AI-assisted,
the AI-detector is evaluated primarily on a separate benchmark of genuinely
human pre-2022 Wikipedia paragraphs (benchmarks/ai_detection/).

Each document is a list of (kind, text) items:
    ("title", ...) ("heading", ...) ("para", ...) ("bullets", [...])
    ("table", [[row], ...]) ("pagebreak", None) ("refs", [...])
"""

TEMPLATE_PROMPT = (
    "Kestrel Foods Ltd. is a fictional mid-sized manufacturer of packaged snacks headquartered in Pune, India. "
    "Over the last three years its revenue grew by eleven percent a year, but its gross margin fell from 38 "
    "percent to 29 percent. Prices of edible oil and potatoes rose sharply, modern-trade retailers demanded "
    "deeper discounts, and two venture-funded direct-to-consumer brands entered its core categories. The board "
    "is considering three options: integrating backward into contract farming, building its own "
    "direct-to-consumer channel, or consolidating its product range. Answer the following questions. First, "
    "diagnose the main drivers of the margin decline. Second, evaluate the direct-to-consumer option against "
    "expansion in modern trade. Third, recommend a three-year plan with clear milestones, key risks and the "
    "metrics the board should track. Use the data pack provided, state every assumption you make, and justify "
    "each recommendation with numbers. Limit your answer to 1,500 words and cite every source you use."
)

TEMPLATE_REFERENCES = [
    "Porter, M. E. (1980). Competitive Strategy: Techniques for Analyzing Industries and Competitors. "
    "New York: Free Press.",
    "Kotler, P., & Keller, K. L. (2016). Marketing Management (15th ed.). Harlow: Pearson.",
    "Kestrel Foods Ltd. (2026). Case Competition Data Pack, Round 1. Organising Committee.",
]


def with_template(team_title, body):
    """Prefix the official case prompt and suffix the shared reference list."""
    return (
        [("title", team_title), ("heading", "Case Prompt (Round 1)"), ("para", TEMPLATE_PROMPT), ("pagebreak", None)]
        + body
        + [("heading", "References"), ("refs", TEMPLATE_REFERENCES)]
    )


# --------------------------------------------------------------------- answer A
ANSWER_A = [
    ("heading", "1. Diagnosis"),
    ("para",
     "Kestrel's problem is not demand. Volumes are still growing at a healthy clip, so the nine point fall in "
     "gross margin has to be explained by cost and price. We traced roughly two thirds of the decline to input "
     "costs. Edible oil and potato prices moved together in 2024 and 2025, and Kestrel had no hedging or "
     "long-term supply contracts, which meant every spike flowed straight into cost of goods sold. The remaining "
     "third came from trade terms. Modern-trade chains now ask for listing fees, promotional funding and longer "
     "credit periods, and Kestrel accepted these terms one account at a time without ever pricing them into its "
     "list prices."),
    ("para",
     "There is also a mix effect that the board papers tend to ignore. The fastest growing products are the "
     "large value packs, and these carry the thinnest margin in the range. In other words the company has been "
     "growing in exactly the wrong place."),
    ("heading", "2. Channel evaluation"),
    ("para",
     "A direct-to-consumer channel looks attractive on paper because it removes the retailer's cut. We think it "
     "is the wrong first move. Customer acquisition costs for snack brands sold online are high, repeat purchase "
     "is driven by impulse rather than loyalty, and the average order value of a snack basket rarely covers "
     "last-mile delivery. The two funded challengers are burning cash to buy share, and Kestrel should not join "
     "a price war it cannot finance."),
    ("para",
     "Modern trade, by contrast, is where Kestrel already has shelf presence. The issue there is negotiating "
     "discipline rather than the channel itself. A simple account-level profitability review would show which "
     "chains destroy value once promotions and credit costs are included."),
    ("heading", "3. Recommendation"),
    ("para",
     "Our three-year plan has three steps. In year one, Kestrel should sign forward contracts for edible oil "
     "covering at least half of its annual requirement and pilot contract farming for potatoes with two farmer "
     "producer organisations. In year two, it should cut the long tail of low-margin stock keeping units by "
     "about a quarter and reprice the value packs. In year three, once margins are back above 34 percent, it can "
     "test a small direct-to-consumer storefront focused on premium gift packs, where order values are higher."),
    ("para",
     "The board should track gross margin by product family, trade spend as a share of revenue, and the share "
     "of raw material volume that is contracted in advance. The main risk is a poor monsoon year that pushes "
     "farm prices above the contracted level, which is why the pilot must start small."),
]

# --------------------------------------------------------------------- answer B
ANSWER_B = [
    ("heading", "1. What is driving the margin decline"),
    ("para",
     "Kestrel's gross margin erosion is mainly a procurement problem. The company buys edible oil and potatoes "
     "on the spot market, so when commodity prices surged it absorbed the full increase without any contractual "
     "protection. Retailer power made things worse: large supermarket chains extracted bigger promotional "
     "allowances while Kestrel kept its list prices flat for fear of losing shelf space. Finally, the product "
     "portfolio has become bloated, with more than one hundred and twenty variants, many of which sell in small "
     "volumes and complicate production planning."),
    ("heading", "2. Direct-to-consumer versus modern trade"),
    ("para",
     "Selling directly to consumers would give Kestrel valuable first-party data and higher realised prices per "
     "pack. However, the economics of delivering low-value snack orders are poor, and building a digital brand "
     "requires sustained marketing spend that the current margin cannot support. Modern trade offers scale and "
     "visibility immediately. We therefore recommend strengthening the modern-trade relationship through joint "
     "business plans rather than launching a separate online channel now."),
    ("heading", "3. Three-year roadmap"),
    ("para",
     "Year one should focus on procurement: Kestrel should negotiate annual price agreements with oil refiners "
     "and start a contract farming programme with potato growers in Maharashtra. Year two should focus on "
     "simplification: the company should remove the slowest-selling variants and redirect marketing money to "
     "its five best-selling brands. Year three should focus on growth: with healthier margins, Kestrel can enter "
     "new states through distributors and revisit the online channel. Key metrics include gross margin, "
     "procurement cost per kilogram, revenue per variant and retailer fill rate. The biggest risk is that "
     "farmers break contracts when spot prices rise above the agreed price."),
]

# ------------------------------------------------- answer B paraphrased (B')
ANSWER_B_PARAPHRASE = [
    ("title", "Kestrel Foods - Executive Summary (Team Charlie)"),
    ("heading", "Why profitability has slipped"),
    ("para",
     "The decline in Kestrel's gross margin stems largely from the way it sources raw materials. Because the "
     "firm purchases cooking oil and potatoes at prevailing market rates, it had no contractual shield when "
     "commodity costs spiked and therefore bore the entire increase. The bargaining strength of retailers "
     "compounded the damage: big grocery chains demanded larger promotional payments, yet Kestrel left its "
     "catalogue prices unchanged out of fear of losing its shelf space. On top of this, the range has grown "
     "unwieldy, with over 120 product variants, a large number of which move in tiny quantities and make "
     "manufacturing schedules harder to manage."),
    ("heading", "Online direct sales or modern retail?"),
    ("para",
     "A direct-to-consumer route would hand Kestrel useful customer data of its own and better price "
     "realisation on every pack. Yet shipping inexpensive snack orders to homes is uneconomic, and creating an "
     "online brand demands continuous advertising investment that today's margins simply cannot fund. Modern "
     "retail, in contrast, delivers reach and visibility straight away. Our advice is to deepen ties with "
     "supermarket partners through jointly agreed business plans instead of opening a standalone web channel at "
     "this stage."),
    ("pagebreak", None),
    ("heading", "A roadmap for the next three years"),
    ("para",
     "In the first year the priority is purchasing: Kestrel ought to agree yearly price contracts with oil "
     "processors and launch a contract-farming scheme with potato farmers in Maharashtra. The second year is "
     "about streamlining: the business should discontinue its weakest-selling variants and channel advertising "
     "budgets towards its five strongest brands. The third year is about expansion: with margins restored, the "
     "company can move into new states via distribution partners and reconsider selling online. Important "
     "indicators are gross margin, sourcing cost per kilo, sales per variant and retailer fill rate. The largest "
     "danger is growers walking away from their agreements whenever market prices climb above the contracted "
     "rate."),
]

# --------------------------------------------------------------------- answer C
ANSWER_C_CHANNEL_PARAGRAPH = (
    "We looked at the direct-to-consumer option through the lens of customer lifetime value. A typical snack "
    "shopper buys small packs on impulse, spends less than three hundred rupees per order, and is unlikely to "
    "subscribe. Under realistic assumptions the payback period on paid acquisition is more than two years, which "
    "is too long for a company whose margins are already under pressure."
)

ANSWER_C = [
    ("heading", "1. Diagnosis"),
    ("para",
     "Most teams will blame commodity prices, and they matter, but we think the deeper issue is pricing power. "
     "Kestrel has spent a decade competing on value. When oil prices rose, rivals with stronger brands passed the "
     "increase on to shoppers within one quarter, while Kestrel waited almost a year and then raised prices by "
     "only four percent. Our estimate is that this delay alone cost about three points of gross margin. The rest "
     "is explained by heavier trade promotions and a shift toward low-priced multipacks."),
    ("heading", "2. Channel evaluation"),
    ("para", ANSWER_C_CHANNEL_PARAGRAPH + " Modern trade remains the better channel, provided Kestrel uses it to "
     "sell premium lines rather than discount packs."),
    ("heading", "3. Recommendation"),
    ("para",
     "We recommend a premiumisation strategy built around a new baked range called Kestrel Gold. Premium snacks "
     "are growing twice as fast as the overall category in urban India and carry margins about ten points "
     "higher. Kestrel should fund the launch by reducing blanket discounts and shifting that money into targeted "
     "in-store sampling. Over three years, the target is to move premium products from eight percent to twenty "
     "five percent of revenue."),
    ("para",
     "The board should monitor price realisation per kilogram, premium share of sales, and promotion return on "
     "investment. The main risk is that shoppers trade down during an economic slowdown, so the core value range "
     "must stay competitive."),
]

# ------------------------------------------------------------ AI-generated (LLM)
AI_GENERATED = [
    ("title", "Strategic Recommendations for Kestrel Foods Ltd."),
    ("heading", "Executive Summary"),
    ("para",
     "Kestrel Foods Ltd. faces a critical juncture in its growth journey. While the company has achieved "
     "impressive revenue growth, its declining gross margin highlights the need for a comprehensive strategic "
     "realignment. This analysis examines the key drivers of margin erosion, evaluates the direct-to-consumer "
     "opportunity, and proposes a holistic three-year roadmap designed to restore profitability while "
     "positioning the company for sustainable long-term growth."),
    ("heading", "Key Drivers of Margin Decline"),
    ("para",
     "Several interconnected factors have contributed to the decline in Kestrel's gross margin. First, rising "
     "raw material costs, particularly for edible oil and potatoes, have significantly increased the cost of "
     "goods sold. Second, the growing bargaining power of modern-trade retailers has resulted in higher "
     "promotional expenses and deeper discounts. Third, intensified competition from well-funded "
     "direct-to-consumer brands has put additional pressure on pricing. Together, these factors have created a "
     "challenging environment that requires a multifaceted response."),
    ("heading", "Evaluating the Direct-to-Consumer Opportunity"),
    ("para",
     "A direct-to-consumer channel offers several potential advantages, including higher margins, direct access "
     "to customer data, and greater control over the brand experience. However, it also presents significant "
     "challenges, such as high customer acquisition costs, complex logistics, and the need for substantial "
     "investment in digital capabilities. In contrast, expanding in modern trade allows Kestrel to leverage its "
     "existing relationships and distribution infrastructure. Ultimately, a balanced approach that prioritizes "
     "modern trade while selectively exploring digital channels is likely to deliver the most sustainable "
     "results."),
    ("heading", "Recommended Three-Year Roadmap"),
    ("para",
     "In the first year, Kestrel should focus on stabilizing its cost base by implementing strategic sourcing "
     "initiatives, including long-term supplier agreements and hedging mechanisms. In the second year, the "
     "company should optimize its product portfolio by rationalizing underperforming SKUs and investing in "
     "high-margin premium offerings. In the third year, Kestrel can build on this strong foundation by scaling "
     "digital channels and expanding into new geographic markets."),
    ("heading", "Key Risks and Metrics"),
    ("para",
     "Key risks include commodity price volatility, retailer pushback, and execution challenges. To mitigate "
     "these risks, Kestrel should establish a robust governance framework and track key performance indicators "
     "such as gross margin, customer acquisition cost, and inventory turnover. By taking decisive action today, "
     "Kestrel Foods can transform its current challenges into opportunities and build a resilient, profitable "
     "business for the future."),
]

# ----------------------------------------------- original, same case (hard negative)
ORIGINAL_SAME_CASE = [
    ("title", "Kestrel case - Team Echo"),
    ("heading", "Our take"),
    ("para",
     "Honestly, we think the margin story is partly an accounting story. When we rebuilt the cost of goods sold "
     "from the data pack, factory overheads had grown faster than volume because the Nashik plant runs at about "
     "sixty percent utilisation. Fixed costs spread over too few tonnes make every pack more expensive, and no "
     "amount of procurement cleverness will fix that."),
    ("para",
     "So our first recommendation is boring but important: close the second shift at the older Satara line and "
     "move that volume to Nashik. We estimate this lifts utilisation to above eighty percent and saves close to "
     "two margin points within eighteen months."),
    ("heading", "On channels"),
    ("para",
     "We would not frame it as direct-to-consumer versus modern trade at all. The bigger opportunity is general "
     "trade, the small kirana stores that still sell most snacks in tier-two towns. Kestrel's distributor "
     "network there is patchy, and distributor margins are lower than what supermarkets demand. A van-sales "
     "pilot in three districts of Karnataka would tell the board quickly whether this route works."),
    ("heading", "What we would measure"),
    ("para",
     "Plant utilisation, conversion cost per tonne, working capital days and numeric distribution in kirana "
     "outlets. If utilisation doesn't cross seventy-five percent by the end of year one, the consolidation plan "
     "has failed and should be revisited."),
    ("heading", "Risks"),
    ("para",
     "Union negotiations at Satara could delay the shift closure, and van sales need tight cash controls "
     "because collections happen in the field. We'd start with a single district before scaling up."),
]

# ------------------------------------------------ original, different case
ORIGINAL_OTHER_CASE = [
    ("title", "GreenSpokes Bike Share: Should the Service Expand to Three New Districts?"),
    ("heading", "Summary"),
    ("para",
     "GreenSpokes, a fictional operator, runs 1,200 shared bicycles across central Nagpur. Ridership doubled "
     "after the metro opened, but the fleet now loses about nine percent of its bikes each year to theft and "
     "vandalism. Our analysis suggests expansion should wait until fleet losses fall below four percent."),
    ("heading", "Demand"),
    ("para",
     "Trip data show that most rides start or end within five hundred metres of a metro station, and weekday "
     "mornings account for almost half of all journeys. The three proposed districts are residential and lack a "
     "metro stop, so we expect each new bike there to generate fewer than two trips per day, compared with five "
     "in the centre."),
    ("heading", "Unit economics"),
    ("para",
     "At the current fare of fifteen rupees per half hour, a bike needs roughly three trips a day to cover "
     "maintenance, rebalancing trucks and depreciation. The new districts would therefore run at a loss unless "
     "the city provides a subsidy or allows advertising on docking stations."),
    ("pagebreak", None),
    ("heading", "Recommendation"),
    ("para",
     "First, invest in GPS-enabled smart locks, which pilots in other cities suggest can halve theft. Second, "
     "add docking stations around the two newest metro stations, where demand is already proven. Third, revisit "
     "the residential districts in eighteen months with a feeder model timed to metro departures."),
    ("heading", "Risks and measures"),
    ("para",
     "The main risk is that the smart-lock vendor cannot deliver at scale. We would track trips per bike per "
     "day, the annual fleet loss rate and the rebalancing cost per trip."),
]

# --------------------------------------------- external copy (Wikipedia, uncited)
WIKI_VERTICAL_P1 = (
    "In microeconomics, management, and international political economy, vertical integration is a term that "
    "describes the arrangement in which the supply chain of a company is integrated and owned by that company. "
    "Usually each member of the supply chain produces a different product or (market-specific) service, and the "
    "products combine to satisfy a common need."
)
WIKI_VERTICAL_P3 = (
    "Vertical integration is often closely associated with vertical expansion which, in economics, is the growth "
    "of a business enterprise through the acquisition of companies that produce the intermediate goods needed by "
    "the business or help market and distribute its product. Such expansion is desired because it secures the "
    "supplies needed by the firm to produce its product and the market needed to sell the product."
)
# Lightly edited from the "Gross margin" article (paragraph 1).
WIKI_GROSS_MARGIN_EDITED = (
    "Gross margin is the difference between revenue and the cost of goods sold, divided by revenue, and it is "
    "always expressed as a percentage."
)

EXTERNAL_COPY = [
    ("title", "Kestrel Foods: The Case for Backward Integration (Team Golf)"),
    ("heading", "Background"),
    ("para", WIKI_GROSS_MARGIN_EDITED + " For Kestrel this ratio has fallen by nine points in three years, which "
     "is far worse than the category average reported in the data pack."),
    ("heading", "Why integrate backward"),
    ("para", WIKI_VERTICAL_P1),
    ("para", WIKI_VERTICAL_P3),
    ("para",
     "For Kestrel, owning or contracting the potato supply would stabilise the single largest input cost. We "
     "propose a joint venture with a farmer producer organisation near Indore, where the company would provide "
     "seed, cold storage and a guaranteed buy-back price in exchange for exclusive supply."),
    ("heading", "Plan and risks"),
    ("para",
     "The joint venture should start with eight hundred acres in the first season and scale to three thousand "
     "acres by year three if yields meet targets. The board should watch yield per acre, the landed cost of "
     "potatoes and the share of supply under contract. The main risk is crop failure, which crop insurance and a "
     "diversified sourcing region can partly offset."),
]

# ----------------------------------------- heavily formatted original (PDF)
FORMATTED_HEADER = "KESTREL FOODS CASE COMPETITION 2026  |  TEAM HOTEL  |  CONFIDENTIAL"
FORMATTED_ORIGINAL = [
    ("title", "SKU Rationalisation and Price-Pack Architecture"),
    ("heading", "Situation"),
    ("para",
     "We started from the sales ledger rather than from strategy frameworks. Out of 124 active SKUs, just "
     "nineteen generate eighty percent of gross profit, while the bottom forty contribute less than three "
     "percent and still consume a quarter of all production changeovers on the fryer lines."),
    ("table", [
        ["Year", "Revenue (INR cr)", "Gross margin", "Active SKUs"],
        ["2023", "412", "38%", "96"],
        ["2024", "458", "33%", "111"],
        ["2025", "509", "29%", "124"],
    ]),
    ("para",
     "The table makes the pattern obvious: every year the range got wider, and every year the margin got "
     "thinner. Complexity has a cost that never shows up as a separate line in the management accounts."),
    ("heading", "What we recommend"),
    ("bullets", [
        "Delist the bottom forty SKUs over two quarters, starting with flavours that duplicate a bestseller.",
        "Introduce three price points only, at twenty, fifty and ninety-nine rupees, so retailers can plan shelves.",
        "Move promotional money from blanket discounts to bundle offers that lift the average basket.",
    ]),
    ("pagebreak", None),
    ("heading", "Financial impact"),
    ("para",
     "Fewer changeovers raise effective fryer capacity by roughly twelve percent, which postpones the planned "
     "new line by at least two years. Combined with the cleaner price ladder, we expect gross margin to recover "
     "to about thirty-three percent by the end of the second year, without relying on any change in commodity "
     "prices."),
    ("heading", "Risks"),
    ("para",
     "Some retailers may resist delisting because they earn listing fees on every SKU. A small number of loyal "
     "regional customers will also lose their favourite flavour, so the delisting should be communicated "
     "through distributors well in advance."),
    ("pagebreak", None),
    ("heading", "Metrics and governance"),
    ("para",
     "A monthly complexity dashboard should report gross profit per SKU, changeover hours, and the share of "
     "revenue from the top twenty products. Any new SKU proposal must show a payback within twelve months "
     "before it reaches the board."),
]

# -------------------------------------------------- answer D (template, original)
ANSWER_D = [
    ("heading", "1. Diagnosis"),
    ("para",
     "Our reading of the data pack is that Kestrel lost margin because it stayed a domestic, single-channel "
     "business while its costs became global. Edible oil is priced off international palm and sunflower "
     "benchmarks, so a currency swing hurts Kestrel immediately, but none of its revenue is earned in dollars to "
     "offset that exposure."),
    ("heading", "2. Channels"),
    ("para",
     "We would not spend scarce capital on a consumer website. Instead, Kestrel's idle shifts could be sold to "
     "large retailers as private-label manufacturing. Retail chains want reliable local suppliers for their own "
     "snack brands, and contract volumes would absorb fixed costs without any marketing spend."),
    ("heading", "3. Three-year plan"),
    ("para",
     "In the first year, sign two private-label contracts and register with the export promotion council. In the "
     "second year, ship ethnic snacks to Gulf markets through an established importer in Dubai, where the Indian "
     "diaspora already knows the products. By the third year, export and private-label revenue should reach a "
     "fifth of the total, which creates a natural hedge against rupee depreciation. We would track capacity "
     "utilisation, export revenue share and the foreign-exchange gain or loss each quarter. The main risk is "
     "that private-label buyers squeeze prices once Kestrel depends on them, so no single buyer should exceed "
     "ten percent of volume."),
]

# ----------------------------------------------------------------- video scripts
PITCH_01_SCRIPT = (
    "Hello, here is our pitch for the Kestrel case. " + ANSWER_C_CHANNEL_PARAGRAPH + " Thank you."
)
WIKI_SCM_P1 = (
    "In commerce, supply chain management is the management of the flow of goods and services and includes all "
    "processes that transform raw materials into final products. This can include the movement and storage of "
    "raw materials, work-in-process inventory, finished goods, and end to end order fulfilment from the point of "
    "origin to the point of consumption."
)
PITCH_02_SCRIPT = "Good morning judges. Before our recommendation, a quick definition. " + WIKI_SCM_P1


DOCUMENTS = {
    "submission_01.pdf": with_template("Kestrel Foods: Fixing the Margin Before Chasing Growth (Team Alpha)", ANSWER_A),
    "submission_02.docx": with_template(
        "Kestrel Foods: Fixing the Margin Before Chasing Growth (Team Sigma)", ANSWER_A),
    "submission_03.docx": with_template("Kestrel Foods Case Response (Team Bravo)", ANSWER_B),
    "submission_04.docx": ANSWER_B_PARAPHRASE,
    "submission_05.pdf": with_template("Kestrel Foods: Premiumise, Don't Discount (Team Delta)", ANSWER_C),
    "submission_06.pdf": AI_GENERATED,
    "submission_07.pdf": ORIGINAL_SAME_CASE,
    "submission_08.docx": ORIGINAL_OTHER_CASE,
    "submission_09.pdf": EXTERNAL_COPY,
    "submission_10.pdf": FORMATTED_ORIGINAL,
    "submission_11.docx": with_template("Kestrel Foods Case Response (Team India)", ANSWER_D),
}
