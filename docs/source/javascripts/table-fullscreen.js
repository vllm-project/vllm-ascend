(function () {
  const labels = document.documentElement.lang.toLowerCase().startsWith('zh')
    ? { open: '全屏查看表格', close: '退出全屏' }
    : { open: 'View table in fullscreen', close: 'Exit fullscreen' }
  const tables = new Map()
  let dialog
  let scrollArea
  let openButton
  let updatePending = false

  function updateOverflow() {
    updatePending = false
    if (dialog && dialog.open) {
      return
    }

    const states = Array.from(tables, ([table, { container, button }]) => ({
      button,
      overflowing: container.clientWidth > 0 &&
        Math.max(table.scrollWidth, container.scrollWidth) > container.clientWidth,
    }))
    states.forEach(({ button, overflowing }) => {
      button.hidden = !overflowing
    })
  }

  function scheduleUpdate() {
    if (!updatePending) {
      updatePending = true
      window.requestAnimationFrame(updateOverflow)
    }
  }

  const observer = new ResizeObserver(scheduleUpdate)

  function createDialog() {
    dialog = document.createElement('dialog')
    dialog.className = 'table-fullscreen-dialog md-typeset'
    dialog.setAttribute('aria-label', labels.open)

    const toolbar = document.createElement('div')
    toolbar.className = 'table-fullscreen-toolbar'
    const closeButton = document.createElement('button')
    closeButton.type = 'button'
    closeButton.className = 'table-fullscreen-button'
    closeButton.setAttribute('aria-label', labels.close)
    closeButton.title = labels.close
    closeButton.textContent = '×'
    closeButton.autofocus = true
    closeButton.addEventListener('click', () => dialog.close())
    toolbar.append(closeButton)

    scrollArea = document.createElement('div')
    scrollArea.className = 'table-fullscreen-scroll'
    scrollArea.tabIndex = 0
    scrollArea.setAttribute('role', 'region')
    scrollArea.setAttribute('aria-label', labels.open)
    dialog.append(toolbar, scrollArea)
    dialog.addEventListener('close', () => {
      scrollArea.replaceChildren()
      document.documentElement.classList.remove('table-fullscreen-active')
      updateOverflow()
      if (openButton && openButton.isConnected && !openButton.hidden) {
        openButton.focus({ preventScroll: true })
      }
      openButton = undefined
    })
    document.body.append(dialog)
  }

  function markFirstColumn(table) {
    for (const section of table.children) {
      if (!section.rows) {
        continue
      }
      let remainingRows = 0
      for (const row of section.rows) {
        if (remainingRows > 0) {
          remainingRows--
          continue
        }
        const cell = row.cells[0]
        if (cell) {
          cell.classList.add('table-fullscreen-first-column')
          // rowSpan=0 extends to the end of this row group.
          remainingRows = cell.rowSpan === 0 ? section.rows.length : cell.rowSpan - 1
        }
      }
    }
  }

  function showTable(table, button) {
    if (!dialog) {
      createDialog()
    }
    if (dialog.open) {
      return
    }

    const clone = table.cloneNode(true)
    if (!table.closest('[data-table-sticky-column="false"]')) {
      markFirstColumn(clone)
    }
    // Keep IDs and their table-local references unique in the same document.
    const ids = new Map()
    const elements = [clone, ...clone.querySelectorAll('*')]
    elements.forEach(element => {
      if (element.id) {
        const id = `table-fullscreen-${element.id}`
        ids.set(element.id, id)
        element.id = id
      }
    })
    elements.forEach(element => {
      for (const attribute of ['headers', 'aria-labelledby', 'aria-describedby', 'for']) {
        if (element.hasAttribute(attribute)) {
          element.setAttribute(attribute, element.getAttribute(attribute)
            .split(/\s+/).map(id => ids.get(id) || id).join(' '))
        }
      }
      const href = element.getAttribute('href')
      if (href && href.startsWith('#') && ids.has(href.slice(1))) {
        element.setAttribute('href', `#${ids.get(href.slice(1))}`)
      }
    })

    scrollArea.replaceChildren(clone)
    openButton = button
    dialog.showModal()
    document.documentElement.classList.add('table-fullscreen-active')
    scrollArea.scrollTo(0, 0)
  }

  function init() {
    // Reuse controls on repeated document$ notifications; drop detached tables.
    observer.disconnect()
    tables.forEach(({ button }, table) => {
      if (!table.isConnected) {
        button.remove()
        tables.delete(table)
      }
    })
    document.querySelectorAll('.md-content table').forEach(table => {
      if (!tables.has(table)) {
        const container = table.closest('.md-typeset__scrollwrap') || table
        const button = document.createElement('button')
        button.type = 'button'
        button.className = 'table-fullscreen-button table-fullscreen-open'
        button.hidden = true
        button.setAttribute('aria-label', labels.open)
        button.title = labels.open
        button.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true">' +
          '<path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>' +
          '</svg>'
        button.addEventListener('click', () => showTable(table, button))
        // Anchor the action outside the scroller so horizontal scrolling keeps it visible.
        const wrapper = document.createElement('div')
        wrapper.className = 'table-fullscreen-wrapper'
        container.before(wrapper)
        wrapper.append(container, button)
        tables.set(table, { container, button })
      }
      observer.observe(table)
      observer.observe(tables.get(table).container)
    })
    scheduleUpdate()
  }

  document$.subscribe(init)
})()
