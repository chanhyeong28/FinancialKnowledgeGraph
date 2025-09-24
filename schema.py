from re import L


ENTITIES = [
  {
    "label": "Instrument",
    "description": "Tradable asset: equity, commodity, FX pair, index, ETF, future",
    "properties": [
      {"name": "instrument_type", "type": "ENUM", "allowed": ["Equity","Commodity","FX","Index","ETF","Future","Option"]},
      {"name": "ticker", "type": "STRING"},
      {"name": "mic", "type": "STRING"},
      {"name": "currency", "type": "STRING"},
      {"name": "sector", "type": "STRING"},
      {"name": "contract_spec", "type": "STRING"},
      {"name": "reference_id", "type": "STRING"}
    ]
  },
  {
    "label": "Company",
    "description": "Issuer or operating entity",
    "properties": [
      {"name": "name", "type": "STRING"},
      {"name": "isin", "type": "STRING"},
      {"name": "country", "type": "STRING"},
      {"name": "gics", "type": "STRING"}
    ]
  },
  {
    "label": "EconomicIndicator",
    "description": "Macro time series used for forecasting",
    "properties": [
      {"name": "indicator_name", "type": "STRING"},
      {"name": "freq", "type": "ENUM", "allowed": ["D","W","M","Q"]},
      {"name": "unit", "type": "STRING"},
      {"name": "seasonal_adjustment", "type": "ENUM", "allowed": ["SA","NSA"]},
      {"name": "region", "type": "STRING"}
    ]
  },
  {
    "label": "IndicatorPrint",
    "description": "A specific release/print of an indicator",
    "properties": [
      {"name": "release_time", "type": "DATETIME"},
      {"name": "period", "type": "STRING"},
      {"name": "actual", "type": "FLOAT"},
      {"name": "consensus", "type": "FLOAT"},
      {"name": "prior", "type": "FLOAT"},
      {"name": "surprise", "type": "FLOAT"},
      {"name": "market_immediate_move", "type": "FLOAT"},
      {"name": "move_window_min", "type": "INTEGER"},
      {"name": "move_window_max", "type": "INTEGER"}
    ]
  },
  {
    "label": "Event",
    "description": "Discrete exogenous driver (geopolitics, supply shock, disaster)",
    "properties": [
      {"name": "event_type", "type": "ENUM", "allowed": ["Geopolitical","NaturalDisaster","IndustrialAccident","Earnings","Guidance","Policy","OPEC","Strike","Sanction","Regulatory","War","Ceasefire"]},
      {"name": "start_time", "type": "DATETIME"},
      {"name": "end_time", "type": "DATETIME"},
      {"name": "location", "type": "STRING"},
      {"name": "status", "type": "ENUM", "allowed": ["Anticipated","Ongoing","Occurred","Resolved"]},
      {"name": "reported_time", "type": "DATETIME"}
    ]
  },
  {
    "label": "PolicyAction",
    "description": "Central bank or regulator decision",
    "properties": [
      {"name": "authority", "type": "STRING"},
      {"name": "action", "type": "ENUM", "allowed": ["Hike","Cut","Hold","QE","QT","Tariff","Ban","Quota"]},
      {"name": "size_bps", "type": "FLOAT"},
      {"name": "statement_text", "type": "STRING"}
    ]
  },
  {
    "label": "Sentiment",
    "description": "Market mood snapshot or series",
    "properties": [
      {"name": "scope", "type": "ENUM", "allowed": ["Instrument","Sector","Market","Country"]},
      {"name": "value", "type": "FLOAT"},
      {"name": "scale", "type": "STRING"},
      {"name": "method", "type": "ENUM", "allowed": ["NewsLLM","Twitter","Forum","Options","Survey","COT"]},
      {"name": "as_of", "type": "DATETIME"},
      {"name": "polarity", "type": "ENUM", "allowed": ["Bearish","Neutral","Bullish"]},
      {"name": "confidence", "type": "FLOAT"}
    ]
  },
  {
    "label": "CommodityFundamental",
    "description": "Flow/stock metrics for commodities",
    "properties": [
      {"name": "metric", "type": "ENUM", "allowed": ["Inventory","Production","Exports","Imports","RigCount","RefineryRuns","SpareCapacity"]},
      {"name": "unit", "type": "STRING"},
      {"name": "value", "type": "FLOAT"},
      {"name": "as_of", "type": "DATETIME"},
      {"name": "region", "type": "STRING"}
    ]
  },
  {
    "label": "FXMacro",
    "description": "Macro state relevant to an FX pair",
    "properties": [
      {"name": "country", "type": "STRING"},
      {"name": "real_rate", "type": "FLOAT"},
      {"name": "terms_of_trade", "type": "FLOAT"},
      {"name": "twin_deficits", "type": "FLOAT"},
      {"name": "as_of", "type": "DATETIME"}
    ]
  },
  {
    "label": "AnalystForecast",
    "description": "Forward estimates for earnings/growth/inflation",
    "properties": [
      {"name": "target_metric", "type": "STRING"},
      {"name": "horizon", "type": "STRING"},
      {"name": "value", "type": "FLOAT"},
      {"name": "as_of", "type": "DATETIME"}
    ]
  },
  {
    "label": "Source",
    "description": "Provenance node",
    "properties": [
      {"name": "source_type", "type": "ENUM", "allowed": ["News","Report","API","Transcript","Manual"]},
      {"name": "publisher", "type": "STRING"},
      {"name": "url", "type": "STRING"},
      {"name": "doc_id", "type": "STRING"}
    ]
  }
]
RELATIONS = [
  {
    "label":"AFFECTS_PRICE_OF",
    "description":"Causal or directional linkage",
    "properties":[
      {"name":"direction","type":"ENUM","allowed":["UP","DOWN","UNKNOWN"]},
      {"name":"lag_days","type":"INTEGER"},
      {"name":"horizon_days","type":"INTEGER"},
      {"name":"magnitude","type":"ENUM","allowed":["LOW","MED","HIGH"]},
      {"name":"confidence","type":"FLOAT"},

      {"name":"tense","type":"ENUM","allowed":["PAST","PRESENT","FUTURE"]},
      {"name":"modality","type":"ENUM",
       "allowed":["FACT","INFERENCE","PREDICTION",
                  "POSSIBLE","PROBABLE","NECESSARY","COUNTERFACTUAL"]},
      {"name":"probability","type":"FLOAT"},
      {"name":"evidence_type","type":"ENUM",
       "allowed":["ObservedMove","HistoricalStat","ModelInference","Guidance","Narrative"]},
      {"name":"hedge_markers","type":"STRING"},          # e.g., "may,might,likely"
      {"name":"effect_window","type":"STRING"},          # e.g., "[-5m,+30m]" or "[0d,+2d]"
      {"name":"realized","type":"ENUM","allowed":["True","False","Unknown"]}
    ]
  },

  {
    "label":"SURPRISE_OF",
    "description":"IndicatorPrint surprise attached to Indicator",
    "properties":[
      {"name":"surprise_std","type":"FLOAT"},
      {"name":"sign","type":"ENUM","allowed":["POS","NEG","ZERO"]},
      {"name":"tense","type":"ENUM","allowed":["PAST","PRESENT"]},
      {"name":"modality","type":"ENUM","allowed":["FACT","INFERENCE"]},
      {"name":"probability","type":"FLOAT"}
    ]
  },

  {
    "label":"DRIVES_SENTIMENT",
    "description":"Event/Indicator influences Sentiment",
    "properties":[
      {"name":"scope","type":"STRING"},
      {"name":"lag_days","type":"INTEGER"},
      {"name":"confidence","type":"FLOAT"},
      {"name":"tense","type":"ENUM","allowed":["PAST","PRESENT","FUTURE"]},
      {"name":"modality","type":"ENUM",
       "allowed":["FACT","INFERENCE","PREDICTION","POSSIBLE","PROBABLE"]},
      {"name":"probability","type":"FLOAT"}
    ]
  },

  {
    "label":"HAS_EXPOSURE_TO",
    "description":"Company/Instrument exposure to commodity/country",
    "properties":[
      {"name":"exposure_type","type":"ENUM","allowed":["InputCost","Revenue","FX","SupplyChain"]},
      {"name":"exposure_share","type":"FLOAT"},
      {"name":"tense","type":"ENUM","allowed":["PRESENT","PAST"]},
      {"name":"modality","type":"ENUM","allowed":["FACT","INFERENCE"]},
      {"name":"probability","type":"FLOAT"}
    ]
  },

  {
    "label":"HEDGED_BY",
    "description":"Instrument hedged by another",
    "properties":[
      {"name":"hedge_ratio","type":"FLOAT"},
      {"name":"tense","type":"ENUM","allowed":["PRESENT","PAST"]},
      {"name":"modality","type":"ENUM","allowed":["FACT","INFERENCE"]},
      {"name":"probability","type":"FLOAT"}
    ]
  },

  {
    "label":"COINTEGRATED_WITH",
    "description":"Statistical long-run linkage",
    "properties":[
      {"name":"p_value","type":"FLOAT"},
      {"name":"beta","type":"FLOAT"},
      {"name":"window","type":"STRING"},
      {"name":"tense","type":"ENUM","allowed":["PAST","PRESENT"]},
      {"name":"modality","type":"ENUM","allowed":["FACT","INFERENCE"]},
      {"name":"probability","type":"FLOAT"}
    ]
  },

  {
    "label":"LEADS",
    "description":"Lead-lag edge from X to Y",
    "properties":[
      {"name":"lead_days","type":"INTEGER"},
      {"name":"method","type":"ENUM","allowed":["Granger","CrossCorr","DomainRule"]},
      {"name":"tense","type":"ENUM","allowed":["PRESENT","FUTURE"]},
      {"name":"modality","type":"ENUM","allowed":["FACT","PREDICTION","PROBABLE"]},
      {"name":"probability","type":"FLOAT"}
    ]
  },

  {
    "label":"SUPPLY_SHOCK_FOR",
    "description":"Event causes supply change for commodity",
    "properties":[
      {"name":"shock","type":"ENUM","allowed":["Increase","Decrease"]},
      {"name":"magnitude","type":"ENUM","allowed":["LOW","MED","HIGH"]},
      {"name":"tense","type":"ENUM","allowed":["PAST","PRESENT","FUTURE"]},
      {"name":"modality","type":"ENUM","allowed":["FACT","INFERENCE","PREDICTION","POSSIBLE","PROBABLE"]},
      {"name":"probability","type":"FLOAT"}
    ]
  },

  {
    "label":"DEMAND_SHOCK_FOR",
    "description":"Event causes demand change",
    "properties":[
      {"name":"shock","type":"ENUM","allowed":["Increase","Decrease"]},
      {"name":"tense","type":"ENUM","allowed":["PAST","PRESENT","FUTURE"]},
      {"name":"modality","type":"ENUM","allowed":["FACT","INFERENCE","PREDICTION","POSSIBLE","PROBABLE"]},
      {"name":"probability","type":"FLOAT"}
    ]
  },

  {
    "label":"REPORTS_ON",
    "description":"Source reports about node",
    "properties":[
      {"name":"as_of","type":"DATETIME"},
      {"name":"tense","type":"ENUM","allowed":["PAST","PRESENT","FUTURE"]},
      {"name":"modality","type":"ENUM","allowed":["FACT","INFERENCE","PREDICTION"]},
      {"name":"probability","type":"FLOAT"}
    ]
  },

  {"label":"ISSUED_BY","description":"Indicator is issued by authority"},
  {"label":"UNDERLYING_OF","description":"Company is underlying of equity/index/ETF"},
  {
    "label":"PEGS",
    "description":"Peg/CB regime constraints",
    "properties":[{"name":"regime","type":"STRING"},
                  {"name":"tense","type":"ENUM","allowed":["PRESENT","PAST"]},
                  {"name":"modality","type":"ENUM","allowed":["FACT","INFERENCE"]},
                  {"name":"probability","type":"FLOAT"}]
  }
]
