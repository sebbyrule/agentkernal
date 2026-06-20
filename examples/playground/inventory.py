"""Sample data + a couple of helpers the example tasks can search and summarize."""

ITEMS = [
    {"sku": "A-100", "name": "widget", "qty": 12, "price": 2.50},
    {"sku": "A-101", "name": "gadget", "qty": 0, "price": 9.99},
    {"sku": "B-200", "name": "sprocket", "qty": 3, "price": 1.25},
    {"sku": "B-201", "name": "cog", "qty": 47, "price": 0.75},
]


def total_value():
    """Total dollar value of everything in stock."""
    return sum(item["qty"] * item["price"] for item in ITEMS)


def out_of_stock():
    """SKUs with zero quantity on hand."""
    # TODO: also warn when qty is below a reorder threshold
    return [item["sku"] for item in ITEMS if item["qty"] == 0]
