-- Defer loading of images and iframes until they are scrolled near the viewport.
-- Only applies to the html format: reveal.js slides manage loading themselves.

if FORMAT ~= "html" then
  return {}
end

local function lazy_iframes(el)
  if el.format == "html" and el.text:find("<iframe") then
    el.text = el.text:gsub("<iframe(%s[^>]*)>", function(attrs)
      if attrs:find("loading=") then
        return "<iframe" .. attrs .. ">"
      end
      return '<iframe loading="lazy"' .. attrs .. ">"
    end)
    return el
  end
end

return {
  {
    Image = function(el)
      if not el.attributes["loading"] then
        el.attributes["loading"] = "lazy"
        return el
      end
    end,
    RawBlock = lazy_iframes,
    RawInline = lazy_iframes,
  }
}
