# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prompt templates for GraphRAG structured extraction and summarization."""

DEFAULT_ENTITY_TYPES: list[str] = [
    "organization",
    "person",
    "geo",
    "event",
]

GRAPHRAG_STRUCTURED_EXTRACTION_PROMPT: str = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify the entities of those types from the text and the relationships among the identified entities. Note that the text document is chunked and may end abruptly.
-Steps-
1. Identify all entities. For each identified entity, extract:
   - name: Name of the entity, capitalized.
   - type: One of the following entity types: [{entity_types}].
   - description: Comprehensive description of the entity's attributes and activities.

2. From the entities identified in step 1, identify pairs of (source_entity, target_entity) that have a *direct and explicit* relationship described in the text.
   - **Avoid Combinatorial Explosion:** Do NOT create relationships between all possible pairs of entities. Only extract relationships that are clearly stated or central to the text.
   - **Avoid Peer Bloat:** If multiple entities share the same relationship to a central entity (e.g., multiple people work at the same organization), link each person to the organization, but do NOT link the people to each other unless a specific relationship between them is mentioned.
   - **Focus on Significance:** Prioritize relationships that convey important actions, ownership, affiliation, or facts. Avoid trivial connections.

For each pair of related entities, extract the following information:
- source: name of the source entity, as identified in step 1
- target: name of the target entity, as identified in step 1
- relation: Clear, concise relationship predicate or explanation of the relationship between source and target.

3. Return the extracted entities and relationships as a structured JSON object. Ensure the output is strictly valid JSON. Do NOT include literal control characters (like newlines, tabs, or carriage returns) within JSON string values unless they are properly escaped.

######################
-Examples-
######################
Example 1:
Allowed Entity Types: ORGANIZATION, PERSON
Text:
The Verdantis Central Institution is scheduled to meet on Monday. Central Institution Chair Martin Smith will take questions at a press conference.

Output:
{{
  "entities": [
    {{
      "name": "CENTRAL INSTITUTION",
      "type": "ORGANIZATION",
      "description": "The Central Institution of Verdantis, responsible for policy decisions."
    }},
    {{
      "name": "MARTIN SMITH",
      "type": "PERSON",
      "description": "Chair of the Central Institution."
    }}
  ],
  "triples": [
    {{
      "source": "MARTIN SMITH",
      "target": "CENTRAL INSTITUTION",
      "relation": "is chair of"
    }}
  ]
}}

Example 3:
Allowed Entity Types: ORGANIZATION, GEO, PERSON
Text:
Five Aurelians jailed for 8 years in Firuzabad and widely regarded as hostages are on their way home to Aurelia.

The swap orchestrated by Quintara was finalized when $8bn of Firuzi funds were transferred to financial institutions in Krohaara, the capital of Quintara.

The exchange initiated in Firuzabad's capital, Tiruzia, led to the four men and one woman, who are also Firuzi nationals, boarding a chartered flight to Krohaara.

They were welcomed by senior Aurelian officials and are now on their way to Aurelia's capital, Cashion.

The Aurelians include 39-year-old businessman Samuel Namara, who has been held in Tiruzia's Alhamia Prison, as well as journalist Durke Bataglani, 59, and environmentalist Meggie Tazbah, 53, who also holds Bratinas nationality.

Output:
{{
  "entities": [
    {{
      "name": "FIRUZABAD",
      "type": "GEO",
      "description": "Firuzabad held Aurelians as hostages"
    }},
    {{
      "name": "AURELIA",
      "type": "GEO",
      "description": "Country seeking to release hostages"
    }},
    {{
      "name": "QUINTARA",
      "type": "GEO",
      "description": "Country that negotiated a swap of money in exchange for hostages"
    }},
    {{
      "name": "TIRUZIA",
      "type": "GEO",
      "description": "Capital of Firuzabad where the Aurelians were being held"
    }},
    {{
      "name": "KROHAARA",
      "type": "GEO",
      "description": "Capital city in Quintara"
    }},
    {{
      "name": "CASHION",
      "type": "GEO",
      "description": "Capital city in Aurelia"
    }},
    {{
      "name": "SAMUEL NAMARA",
      "type": "PERSON",
      "description": "Aurelian who spent time in Tiruzia's Alhamia Prison"
    }},
    {{
      "name": "ALHAMIA PRISON",
      "type": "GEO",
      "description": "Prison in Tiruzia"
    }},
    {{
      "name": "DURKE BATAGLANI",
      "type": "PERSON",
      "description": "Aurelian journalist who was held hostage"
    }},
    {{
      "name": "MEGGIE TAZBAH",
      "type": "PERSON",
      "description": "Bratinas national and environmentalist who was held hostage"
    }}
  ],
  "triples": [
    {{
      "source": "FIRUZABAD",
      "target": "AURELIA",
      "relation": "negotiated hostage exchange with"
    }},
    {{
      "source": "QUINTARA",
      "target": "AURELIA",
      "relation": "brokered hostage exchange for"
    }},
    {{
      "source": "QUINTARA",
      "target": "FIRUZABAD",
      "relation": "brokered hostage exchange with"
    }},
    {{
      "source": "SAMUEL NAMARA",
      "target": "ALHAMIA PRISON",
      "relation": "held as prisoner at"
    }},
    {{
      "source": "SAMUEL NAMARA",
      "target": "MEGGIE TAZBAH",
      "relation": "exchanged together in hostage release with"
    }},
    {{
      "source": "SAMUEL NAMARA",
      "target": "DURKE BATAGLANI",
      "relation": "exchanged together in hostage release with"
    }},
    {{
      "source": "MEGGIE TAZBAH",
      "target": "DURKE BATAGLANI",
      "relation": "exchanged together in hostage release with"
    }},
    {{
      "source": "SAMUEL NAMARA",
      "target": "FIRUZABAD",
      "relation": "held as hostage in"
    }},
    {{
      "source": "MEGGIE TAZBAH",
      "target": "FIRUZABAD",
      "relation": "held as hostage in"
    }},
    {{
      "source": "DURKE BATAGLANI",
      "target": "FIRUZABAD",
      "relation": "held as hostage in"
    }}
  ]
}}

######################
-Real Data-
######################
Allowed Entity Types: {entity_types}
"""
