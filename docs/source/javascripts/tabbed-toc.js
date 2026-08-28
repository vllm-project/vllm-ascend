(function () {
  function revealTabbedTarget() {
    if (!window.location.hash) {
      return
    }

    const id = decodeURIComponent(window.location.hash.slice(1))
    const target = document.getElementById(id)
    if (!target) {
      return
    }

    let block = target.closest('.tabbed-block')
    while (block) {
      const content = block.parentElement
      const tabSet = content && content.parentElement
      if (!tabSet || !tabSet.classList.contains('tabbed-set')) {
        break
      }

      const blocks = Array.from(
        tabSet.querySelectorAll(':scope > .tabbed-content > .tabbed-block')
      )
      const index = blocks.indexOf(block)
      const inputs = tabSet.querySelectorAll(':scope > input')
      if (index >= 0 && inputs[index] && !inputs[index].checked) {
        inputs[index].click()
      }

      block = tabSet.closest('.tabbed-block')
    }

    window.requestAnimationFrame(() => {
      target.scrollIntoView({ block: 'start' })
    })
  }

  document$.subscribe(revealTabbedTarget)
  window.addEventListener('hashchange', revealTabbedTarget)
})()
