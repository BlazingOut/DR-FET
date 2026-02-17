import json
from collections import defaultdict, Counter

class TagManager:
    def __init__(self):
        self.tag_tree = {}  # Pure tag hierarchy structure
        self.tag_index = {}  # Complete tag info (including descriptions)
        self.all_tags = set()  # Flat set of all tags

    def save_to_json(self, file_path):
        """Save tag data to JSON file"""
        data = {
            "tag_tree": self.tag_tree,
            "tag_index": self.tag_index,
            "all_tags": list(self.all_tags)
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_json(cls, file_path):
        """Load tag data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        manager = cls()
        manager.tag_tree = data["tag_tree"]
        manager.tag_index = data.get("tag_index", {})
        manager.all_tags = set(data.get("all_tags", []))
        return manager

    def build_tag_hierarchy(self, tags):
        """
        Build tag hierarchy structure without description info

        Args:
            tags: List of tags, e.g., ['/person/artist', '/organization/sports_team']
        """
        self.all_tags.update(tags)
        self.tag_tree = {}

        for tag in tags:
            if not tag.startswith('/'):
                continue

            parts = [p for p in tag.split('/') if p]
            current_level = self.tag_tree

            # Build hierarchy structure
            for i, part in enumerate(parts):
                if part not in current_level:
                    current_level[part] = {"children": {}}

                # Record full path to all_tags
                full_path = '/' + '/'.join(parts[:i + 1])
                self.all_tags.add(full_path)

                current_level = current_level[part]["children"]

    def get_parent_tag(self, tag):
        """
        Get parent tag

        Args:
            tag: Current tag, e.g., '/person/artist/actor'

        Returns:
            Parent tag path, e.g., '/person/artist' or None
        """
        if tag not in self.all_tags:
            return None

        parts = [p for p in tag.split('/') if p]
        if len(parts) <= 1:
            return None

        parent_path = '/' + '/'.join(parts[:-1])
        return parent_path if parent_path in self.all_tags else None

    def get_child_tags(self, tag):
        """
        Get direct child tags

        Args:
            tag: Current tag

        Returns:
            List of direct child tags
        """
        if tag == '/':
            # Special handling for root
            parts = []
            current_level = self.tag_tree
        else:
            if tag not in self.all_tags:
                return []
            parts = [p for p in tag.split('/') if p]
            current_level = self.tag_tree
            for part in parts:
                if part not in current_level:
                    return []
                current_level = current_level[part]["children"]

        # Collect full paths of direct children
        child_tags = []
        for child_name, child_data in current_level.items():
            child_path = tag + '/' + child_name if tag != '/' else '/' + child_name
            child_tags.append(child_path)

        return child_tags

    def get_all_children(self, tag):
        """
        Get all descendant tags

        Args:
            tag: Current tag

        Returns:
            List of all descendant tags
        """
        children = []
        direct_children = self.get_child_tags(tag)

        for child in direct_children:
            children.append(child)
            children.extend(self.get_all_children(child))

        return children

    def get_sibling_tags(self, tag):
        """
        Get sibling tags (other tags at same level)

        Args:
            tag: Current tag

        Returns:
            List of sibling tags
        """
        parent = self.get_parent_tag(tag)
        if parent is None:
            # Root level, return all first-level tags
            return self.get_child_tags('/')

        siblings = self.get_child_tags(parent)
        return [sib for sib in siblings if sib != tag]

    def get_descriptions(self, tag):
        """Get descriptions for a tag"""
        return self.tag_index[tag]["description"]

    def find_tags_by_pattern(self, pattern):
        """
        Find tags by pattern (simple fuzzy matching)

        Args:
            pattern: Matching pattern, e.g., 'art' matches tags containing 'art'

        Returns:
            List of matching tags
        """
        return [tag for tag in self.all_tags if pattern.lower() in tag.lower()]

    def add_descriptions(self, descriptions_dict):
        """
        Add description info to existing tag structure

        Args:
            descriptions_dict: Dictionary of tag descriptions {tag_path: [descriptions]}
        """
        for tag, descriptions in descriptions_dict.items():
            if tag in self.all_tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = {"description": [], "depth": tag.count('/')}
                self.tag_index[tag]["description"] = descriptions

    def get_tag_info(self, tag):
        """
        Get complete tag info (structure + description)

        Returns:
            Dictionary containing structure and description info
        """
        info = {
            "tag": tag,
            "depth": tag.count('/'),
            "parent": self.get_parent_tag(tag),
            "children": self.get_child_tags(tag),
            "siblings": self.get_sibling_tags(tag)
        }

        if tag in self.tag_index:
            info["description"] = self.tag_index[tag]["description"]
        else:
            info["description"] = []

        return info


def build_label_description_dict(data_list, k=7):
    """
    Build mapping from labels to most frequent descriptive words
    Only updates the highest (most specific) tag in each label path

    Args:
        data_list: List of data, each element is dict with "description" and "label" keys
        k: Number of most frequent descriptive words to keep, default 7

    Returns:
        dict: Key is label, value is list of top k most frequent descriptive words
    """
    from collections import defaultdict, Counter

    # Initialize label-description mapping
    label_description_map = defaultdict(Counter)

    # Process data, count word frequency for each label
    for item in data_list:
        if not item.get("description"):
            continue

        # Get description word list (comma-separated)
        descriptors = [desc.strip() for desc in item["description"].split(",")]

        # Get all labels for this sample
        labels = item["label"]

        # Find highest (most specific) tag in each label path
        # Sort labels by depth (deepest first)
        labels_sorted = sorted(labels, key=lambda x: x.count('/'), reverse=True)

        # Track which labels to update
        labels_to_update = set()

        # Process from deepest labels
        for i, label in enumerate(labels_sorted):
            # Check if this label is prefix of any deeper label
            # If yes, more specific label already includes this label's info
            is_subsumed = False
            for j in range(i):
                if labels_sorted[j].startswith(label + '/'):
                    is_subsumed = True
                    break
            if not is_subsumed:
                labels_to_update.add(label)

        # Only update most specific labels
        for label in labels_to_update:
            label_description_map[label].update(descriptors)

    # Extract top k most frequent descriptive words for each label
    result = {}
    for label, counter in label_description_map.items():
        top_descriptions = [desc for desc, count in counter.most_common(k)]
        result[label] = top_descriptions

    return result

def build_tag_tree(set_name, data_file, save_path):
    type_file = f"data/{set_name}/types.txt"
    with open(type_file) as f:
        label_list = [line.strip() for line in f.readlines()]
    tag_manager = TagManager()
    tag_manager.build_tag_hierarchy(label_list)
    with open(data_file, 'r', encoding='utf-8') as f:
        description_data = [json.loads(line) for line in f.readlines()]
    description_dict = build_label_description_dict(description_data, k=10)
    tag_manager.add_descriptions(description_dict)

    tag_manager.save_to_json(save_path)
    print("done")