You are given a concept word and a list of properties that refer to this word. You have to classify each property into one of the following categories.

**TAXONOMIC** — Characterizes a place in a hierarchy of potential relationships between the feature and the concept, including superordinate and subordinate relationships, as well as individual instances.

Subordinate categories:
- `synonym` — A word with the same meaning as the concept. (e.g., *Radio* → stereo; *Angry* → mad)
- `superordinate` — A category one level above the concept if laid out in a taxonomy. (e.g., *Desk* → furniture)
- `coordinate` — A category at the same level as the concept if laid out in a taxonomy. (e.g., *Sword* → dagger)
- `subordinate` — A category one level below the concept if laid out in a taxonomy. (e.g., *Cheese* → Swiss)
- `individual` — A specific instance of a concept. (e.g., *Monument* → Eiffel Tower; *Atrocity* → September 11)

**ENTITY** — Reflects perceptible and intrinsic features of the concept, including color, shape, texture, size, smell, taste, magnitude, or quantity. Also includes associated abstract entities linked intrinsically with the concept and not specifically associated via contextual co-occurrence. Observable physical expressions of emotion (e.g., "chest puffed out," "wrinkled forehead") are coded as entity properties given their function as surface properties that make emotions interpretable.

Subordinate categories:
- `associated_abstract_entity` — Something that cannot be physically experienced (seen, touched, etc.), but is associated with or co-occurs with the concept. (e.g., *Sword* → honor; *Education* → literacy)
- `entity_behavior` — A typical action a concept performs. (e.g., *Ball* → bounces)
- `external_component` — An external three-dimensional part of the concept. (e.g., *Jacket* → sleeve)
- `external_surface_property` — A property observable from the outside of something, including color, shape, pattern, texture, size, smell, taste, or sound. (e.g., *Sword* → rusty; *Unprepared* → messy hair)
- `internal_component` — A three-dimensional part of the concept that cannot be seen from the outside. (e.g., *Jacket* → down or feathers)
- `internal_surface_property` — A property observable through the senses on the inside of something, including color, shape, pattern, texture, size, smell, taste, or sound. (e.g., *Tomato* → juicy)
- `systemic_property` — An emergent property of the concept, produced through the combination of all of its parts: conditions, abilities, and traits. (e.g., *Jacket* → warm)
- `larger_whole` — The feature represents a larger entity that the concept is part of. (e.g., *Elbow* → part of body)
- `quantity` — Number, frequency, or intensity of the concept. (e.g., *Ear* → come in a pair)
- `made_of` — The feature is the material or thing the concept is made of. (e.g., *Key* → metal)

**SITUATION** — Includes contextual knowledge about a concept's relationships to elements in situations it occurs in. These include the action or manner in which a concept is used, participants in situations involving the concept, the time or location of such situations, and the function of the concept. Social and communicative function is treated as a subtype of function within this category.

Subordinate categories:
- `action_or_manner` — How you use, demonstrate, or interact with the concept. (e.g., *Key* → insert in lock and turn; *Disapproval* → booing)
- `associated_entity` — Another distinct entity that you would find in a situation where the concept occurs. (e.g., *Desk* → chair; *Commitment* → ring)
- `function` — The purpose a concept serves. (e.g., *Jacket* → keeps top warm; *Livelihood* → puts food on the table)
- `location` — Where the concept is found or takes place. (e.g., *Horse* → farm; *Knowledge* → books)
- `origin` — Where the concept comes from. (e.g., *Itch* → bug bites)
- `participant` — A person in a situation who uses the concept, or interacts with other participants in the situation. (e.g., *Sword* → knight; *Adoring* → parents)
- `time` — When a situation involving the concept occurs. (e.g., *Jacket* → winter; *Surprise* → birthday)
- `social_or_communicative_function` — The concept is related to a social relationship or a communicative act. (e.g., *Alcohol* → friends; *Disapproval* → society)

**INTROSPECTIVE** — Reflects personal experiences of the concept. This includes emotional or affective responses, mental states including cognitive operations made by the participant regarding the concept such as comparison or contingencies of the concept, or evaluations of the concept or bodily responses to the concept.

Subordinate categories:
- `affect_or_emotion` — An emotion felt toward the concept or a situation involving the concept. (e.g., *Prize* → happy; *Incest* → disgusting)
- `evaluation` — Positive or negative reactions to the concept, or assessments of how one might feel about the concept. (e.g., *Cheese* → people like this; *Indecision* → a bad thing)
- `cognitive_operation` — Mental comparisons of the concept or one of its properties with other things. (e.g., *Desk* → similar to a counter; *Dislike* → less strong than hate)
- `contingency` — Thinking about something the concept depends on, requires, or is needed to allow to happen. (e.g., *Phone* → you need a data plan)
- `negation` — The feature highlights the absence of something. (e.g., *Radio* → is not a television; *Angry* → not happy)

---

Concept: {word}
Properties: {properties}

Respond in JSON format only:
```json
[
  {
    "property": "<property>",
    "subordinate": "<subordinate_label>",
    "superordinate": "<TAXONOMIC|ENTITY|SITUATION|INTROSPECTIVE>"
  }
]
```